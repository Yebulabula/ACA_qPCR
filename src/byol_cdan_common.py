import copy
import datetime
import json
import logging
import os
from dataclasses import dataclass

import network
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loss import CDAN, DANN, Entropy
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None
    ConfusionMatrixDisplay = None

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


@dataclass
class RunContext:
    dir_out: str
    timestamp: str
    run_output_dir: str
    log_file: str | None


class ContrastiveCurveDataset(Dataset):
    def __init__(self, df_curves, params_csv_path, augment_fn):
        self.df_curves = df_curves
        self.params_df = pd.read_csv(params_csv_path, index_col=0).dropna()
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.df_curves)

    def __getitem__(self, idx):
        c1 = self.augment_fn(self.params_df, idx)
        c2 = self.augment_fn(self.params_df, idx)
        return c1, c2

    def get_original_data(self, idx):
        return self.df_curves.iloc[idx]


def create_run_context(dir_out, use_timestamp_subdir=True, write_log_file=True):
    os.makedirs(dir_out, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(dir_out, timestamp) if use_timestamp_subdir else dir_out
    os.makedirs(run_output_dir, exist_ok=True)
    return RunContext(
        dir_out=dir_out,
        timestamp=timestamp,
        run_output_dir=run_output_dir,
        log_file=os.path.join(run_output_dir, "training.log") if write_log_file else None,
    )


def configure_logging(run_context):
    handlers = [logging.StreamHandler()]
    if run_context.log_file is not None:
        handlers.insert(0, logging.FileHandler(run_context.log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    if run_context.log_file is not None:
        logging.info("Starting BYOL_CDAN training. Logs saved to: %s", run_context.log_file)
    else:
        logging.info("Starting BYOL_CDAN training. File logging disabled.")


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, type):
        return value.__name__
    return value


def is_wandb_enabled(config):
    return config.get("wandb", {}).get("enabled", False) and wandb is not None


def maybe_init_wandb(config, args, f_config, run_context, pretrained_byol_checkpoint, pretrained_cl_checkpoint, run_name_prefix):
    if not config["wandb"]["enabled"]:
        logging.info("Weights & Biases logging disabled by configuration.")
        return None

    if wandb is None:
        logging.warning("wandb is not installed. Continuing without Weights & Biases logging.")
        return None

    wandb_config = {
        "args": sanitize_for_json(vars(args)),
        "training": sanitize_for_json(config),
        "model": sanitize_for_json(f_config),
        "timestamp": run_context.timestamp,
        "pretrained_byol_checkpoint": pretrained_byol_checkpoint,
        "pretrained_cl_checkpoint": pretrained_cl_checkpoint,
    }

    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity") or None,
        name=config["wandb"].get("run_name") or f"{run_name_prefix}-{run_context.timestamp}",
        mode=config["wandb"].get("mode", "online"),
        dir=run_context.run_output_dir,
        config=wandb_config,
        tags=config["wandb"].get("tags", []),
        notes=config["wandb"].get("notes"),
        reinit=True,
    )
    logging.info("Initialized wandb run: %s", run.name)
    if getattr(run, "url", None):
        logging.info("W&B run URL: %s", run.url)
    return run


def maybe_log_wandb(config, metrics, step=None, commit=True):
    if is_wandb_enabled(config):
        wandb.log(metrics, step=step, commit=commit)


def maybe_log_wandb_artifact(config, key, file_path):
    if is_wandb_enabled(config) and os.path.exists(file_path):
        wandb.log({key: wandb.Image(file_path)})


def save_json(file_path, payload):
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, indent=2)


def save_checkpoint(model, path, config, run_context, model_name, best_metric=None, step=None, extra_state=None):
    checkpoint = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "config": sanitize_for_json(config),
        "timestamp": run_context.timestamp,
        "model_name": model_name,
        "best_metric": best_metric,
        "step": step,
    }
    if extra_state:
        checkpoint.update(sanitize_for_json(extra_state))
    torch.save(checkpoint, path)
    logging.info("Checkpoint saved to: %s", path)

def extract_model_for_saving(model):
    if isinstance(model, nn.Sequential) and len(model) == 1:
        return model[0]
    return model


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=0, min_lr_scale=0.01):
    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), total_steps - 1)

    def lr_lambda(step):
        current_step = step + 1
        if warmup_steps > 0 and current_step <= warmup_steps:
            return current_step / warmup_steps

        if total_steps <= warmup_steps:
            return 1.0

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def eval_best_model(best_model, data_loaders, activities, save_name, device, run_context, mode="test", config=None, step=None):
    logging.info("-------Curve-level performance evaluation-------")
    best_model.eval()
    best_model.to(device)
    pred_list = torch.tensor([], device=device)
    label_list = torch.tensor([], device=device)

    with torch.no_grad():
        for data, label in data_loaders[mode]:
            data, label = data.to(device), label.to(device)
            _, output, _ = best_model(data)
            pred = output.data.max(1, keepdim=True)[1]
            pred_list = torch.cat((pred_list, torch.flatten(pred)), 0)
            label_list = torch.cat((label_list, torch.flatten(label)), 0)

    report = classification_report(
        label_list.cpu().numpy(),
        pred_list.cpu().numpy(),
        digits=5,
        target_names=activities,
    )
    logging.info("\n%s", report)

    report_dict = classification_report(
        label_list.cpu().numpy(),
        pred_list.cpu().numpy(),
        target_names=activities,
        output_dict=True,
        zero_division=0,
    )

    y_true = label_list.cpu().numpy()
    y_pred = pred_list.cpu().numpy()

    cm_path = None
    if plt is not None and ConfusionMatrixDisplay is not None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=activities)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {save_name}")
        cm_path = os.path.join(run_context.run_output_dir, f"confusion_matrix_{save_name}_{run_context.timestamp}.png")
        plt.savefig(cm_path)
        plt.close()
        logging.info("Confusion matrix saved to: %s", cm_path)
    else:
        logging.warning("matplotlib is not installed; skipping confusion matrix plot.")

    accuracy = 100.0 * (y_pred == y_true).sum() / len(y_true)
    metrics = {
        "accuracy": accuracy,
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "classification_report": report_dict,
        "confusion_matrix_path": cm_path,
    }

    if config is not None:
        prefix = f"{save_name.lower()}/{mode}"
        maybe_log_wandb(
            config,
            {
                f"{prefix}/accuracy": metrics["accuracy"],
                f"{prefix}/macro_precision": metrics["macro_precision"],
                f"{prefix}/macro_recall": metrics["macro_recall"],
                f"{prefix}/macro_f1": metrics["macro_f1"],
                f"{prefix}/weighted_precision": metrics["weighted_precision"],
                f"{prefix}/weighted_recall": metrics["weighted_recall"],
                f"{prefix}/weighted_f1": metrics["weighted_f1"],
            },
            step=step,
        )
        maybe_log_wandb_artifact(config, f"{prefix}/confusion_matrix", cm_path)

    return metrics


def test(model, data_loader, iter_idx, device):
    model.eval()
    test_loss = 0.0
    correct_class = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            _, output, _ = model(data)
            loss = criterion(output, label.long())
            test_loss += loss.item() * data.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct_class += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_loss = test_loss / len(data_loader.dataset)
    accuracy = 100.0 * correct_class.item() / len(data_loader.dataset)
    logging.info("Test iteration %s: Accuracy = %.2f%%", iter_idx, accuracy)

    return {
        "iter": iter_idx,
        "average_loss": test_loss,
        "correct_class": correct_class.item(),
        "total_elems": len(data_loader.dataset),
        "accuracy %": accuracy,
    }


def train_baseline(config, base_network, data_loaders, device, run_context):
    parameter_list = base_network.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    len_train_source = len(data_loaders["source"])
    total_steps = max(1, config["num_iterations"])
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=config["train_scheduler"].get("warmup_steps", 0),
        min_lr_scale=config["train_scheduler"].get("min_lr_scale", 0.01),
    )
    logging.info(
        "Using baseline scheduler: warmup_steps=%s total_steps=%s min_lr_scale=%.4f",
        config["train_scheduler"].get("warmup_steps", 0),
        total_steps,
        config["train_scheduler"].get("min_lr_scale", 0.01),
    )
    best_acc = 0.0
    best_model = None
    training_accuracy = []
    testing_accuracy = []

    for i in range(config["num_iterations"]):
        base_network.train(True)
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            test_target = test(base_network, data_loaders["test"], i, device)
            temp_acc = test_target["accuracy %"]
            temp_model = nn.Sequential(base_network)
            maybe_log_wandb(
                config,
                {
                    "baseline/test_accuracy": float(temp_acc),
                    "baseline/test_loss": float(test_target["average_loss"]),
                },
                step=i + 1,
            )

            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
                model_path = os.path.join(
                    run_context.run_output_dir,
                    config.get("checkpoint_names", {}).get("baseline", "baseline.pth"),
                )
                save_checkpoint(
                    extract_model_for_saving(best_model),
                    model_path,
                    config,
                    run_context,
                    model_name="baseline",
                    best_metric={"accuracy": float(temp_acc)},
                    step=i + 1,
                )
                logging.info("New best baseline model saved to %s with accuracy: %.2f%%", model_path, temp_acc)

            testing_accuracy.append((i + 1, temp_acc))

        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(data_loaders["source"])

        inputs_source, labels_source = next(iter_source)
        inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)
        _, outputs_source, _ = base_network(inputs_source)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        classifier_loss.backward()
        optimizer.step()
        scheduler.step()

        preds = outputs_source.argmax(dim=1)
        train_acc = (preds == labels_source).float().mean().item()
        training_accuracy.append((i + 1, train_acc))
        maybe_log_wandb(
            config,
            {
                "baseline/train_classification_loss": classifier_loss.item(),
                "baseline/train_accuracy": train_acc * 100.0,
                "baseline/lr": get_current_lr(optimizer),
            },
            step=i + 1,
        )

        if (i + 1) % 500 == 0:
            logging.info(
                "[Iter: %s/%s] Classification loss: %.4f",
                i + 1,
                config["num_iterations"],
                classifier_loss.item(),
            )

    logging.info("Baseline training complete! Plotting training curves...")
    train_iters, train_acc = zip(*training_accuracy)
    test_iters, test_acc = zip(*testing_accuracy)
    train_acc_percentage = [x * 100 for x in train_acc]
    plt.figure()
    plt.plot(train_iters, train_acc_percentage, label="Training Accuracy (%)")
    plt.plot(test_iters, test_acc, label="Testing Accuracy (%)")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy Curves - Baseline")
    plt.legend()
    training_curve_path = os.path.join(run_context.run_output_dir, "training_curve_baseline_model.png")
    plt.savefig(training_curve_path)
    plt.close()
    logging.info("Training curve saved to: %s", training_curve_path)
    maybe_log_wandb_artifact(config, "baseline/training_curve", training_curve_path)

    return best_model


def train_CDAN(config, cdan_model, ad_net, random_layer, data_loaders, device, run_context, checkpoint_key="cdan", model_name="cdan"):
    logging.info("Starting %s training...", model_name.upper())
    parameter_list = cdan_model.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))

    len_train_source = len(data_loaders["source"])
    len_train_target = len(data_loaders["target"])
    total_steps = max(1, config["num_iterations"])
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=config["train_scheduler"].get("warmup_steps", 0),
        min_lr_scale=config["train_scheduler"].get("min_lr_scale", 0.01),
    )
    logging.info(
        "Using CDAN scheduler: warmup_steps=%s total_steps=%s min_lr_scale=%.4f",
        config["train_scheduler"].get("warmup_steps", 0),
        total_steps,
        config["train_scheduler"].get("min_lr_scale", 0.01),
    )
    best_acc = 0.0
    best_model = None
    training_accuracy = []
    testing_accuracy = []
    classifier_loss_iter = []
    transfer_loss_iter = []
    total_loss_iter = []

    for i in range(config["num_iterations"]):
        cdan_model.train(True)
        if i % config["test_interval"] == config["test_interval"] - 1:
            cdan_model.train(False)
            logging.info("Testing CDAN model on Train Data...")
            train_eval = test(cdan_model, data_loaders["source"], i, device)
            logging.info("Testing CDAN model on Test Data...")
            test_target = test(cdan_model, data_loaders["test"], i, device)
            temp_acc = test_target["accuracy %"]
            temp_model = nn.Sequential(cdan_model)
            maybe_log_wandb(
                config,
                {
                    "cdan/eval_source_accuracy": float(train_eval["accuracy %"]),
                    "cdan/eval_source_loss": float(train_eval["average_loss"]),
                    "cdan/eval_test_accuracy": float(temp_acc),
                    "cdan/eval_test_loss": float(test_target["average_loss"]),
                },
                step=i + 1,
            )

            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
                model_path = os.path.join(
                    run_context.run_output_dir,
                    config.get("checkpoint_names", {}).get(checkpoint_key, f"{checkpoint_key}.pth"),
                )
                save_checkpoint(
                    extract_model_for_saving(best_model),
                    model_path,
                    config,
                    run_context,
                    model_name=model_name,
                    best_metric={"accuracy": float(temp_acc)},
                    step=i + 1,
                )
                logging.info("New best %s model saved to %s with accuracy: %.2f%%", model_name, model_path, temp_acc)

            testing_accuracy.append((i + 1, temp_acc))

        loss_params = config["loss"]
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(data_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(data_loaders["target"])

        inputs_source, labels_source = next(iter_source)
        inputs_target, _ = next(iter_target)
        inputs_source, inputs_target, labels_source = inputs_source.to(device), inputs_target.to(device), labels_source.to(device)

        features_source, outputs_source, _ = cdan_model(inputs_source)
        features_target, outputs_target, _ = cdan_model(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        if config["method"] == "CDAN+E":
            entropy = Entropy(softmax_out)
            _, transfer_loss = CDAN([features, softmax_out], ad_net, device, entropy, network.calc_coeff(i), random_layer)
        elif config["method"] == "CDAN":
            _, transfer_loss = CDAN([features, softmax_out], ad_net, device, None, None, random_layer)
        elif config["method"] == "DANN":
            _, transfer_loss = DANN(features, ad_net)
        else:
            raise ValueError("Method cannot be recognized.")

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        classifier_loss_iter.append(classifier_loss.item())
        transfer_loss_iter.append(transfer_loss.item())
        total_loss_iter.append(total_loss.item())

        preds = outputs_source.argmax(dim=1)
        train_acc = (preds == labels_source).float().mean().item()
        training_accuracy.append((i + 1, train_acc))
        maybe_log_wandb(
            config,
            {
                "cdan/train_classification_loss": classifier_loss.item(),
                "cdan/train_transfer_loss": transfer_loss.item(),
                "cdan/train_total_loss": total_loss.item(),
                "cdan/train_source_accuracy": train_acc * 100.0,
                "cdan/lr": get_current_lr(optimizer),
            },
            step=i + 1,
        )

        if (i + 1) % 200 == 0:
            logging.info(
                "[Iter: %s/%s] Classification loss: %.4f, CDAN loss: %.4f, Total loss: %.4f",
                i + 1,
                config["num_iterations"],
                classifier_loss.item(),
                transfer_loss.item(),
                total_loss.item(),
            )

    logging.info("CDAN training complete! Plotting curves...")
    train_iters, train_acc = zip(*training_accuracy)
    test_iters, test_acc = zip(*testing_accuracy)
    train_acc_percentage = [x * 100 for x in train_acc]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_iters, train_acc_percentage, label="Training Accuracy (%)")
    plt.plot(test_iters, test_acc, label="Testing Accuracy (%)")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy - CDAN")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(classifier_loss_iter) + 1), classifier_loss_iter, label="Classifier Loss")
    plt.plot(range(1, len(transfer_loss_iter) + 1), transfer_loss_iter, label="Transfer Loss")
    plt.plot(range(1, len(total_loss_iter) + 1), total_loss_iter, label="Total Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("CDAN Training Losses")
    plt.legend()

    plt.tight_layout()
    curves_path = os.path.join(run_context.run_output_dir, "training_curves_CDAN.png")
    plt.savefig(curves_path)
    plt.close()
    logging.info("CDAN training curves saved to: %s", curves_path)
    maybe_log_wandb_artifact(config, "cdan/training_curves", curves_path)

    return best_model


def build_domain_adversary(model, config, device):
    if config["loss"]["random"]:
        random_layer = network.RandomLayer(
            [model.output_num(), config["class_num"]],
            config["loss"]["random_dim"],
            device,
        ).to(device)
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 256).to(device)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(
            model.output_num() * config["class_num"], 256
        ).to(device)
    return random_layer, ad_net


def strip_state_dict_prefixes(state_dict):
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("0.", "", 1) if key.startswith("0.") else key
        new_key = new_key.replace("module.", "", 1) if new_key.startswith("module.") else new_key
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def load_matching_state_dict(model, checkpoint_path, device, strip_prefixes=False):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if strip_prefixes:
        state_dict = strip_state_dict_prefixes(state_dict)

    model_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    model_state.update(filtered_state)
    model.load_state_dict(model_state)
    logging.info("Loaded %s matching tensors from %s", len(filtered_state), checkpoint_path)


def train_BYOL(config, transformer_model, device, run_context, f_config, image_size, cl_slice, params_csv_path, augment_fn):
    from byol_pytorch import BYOL
    from tqdm.auto import tqdm

    df_curves_cl = pd.read_csv(
        os.path.join(config["data"]["dir_data"], config["data"]["CL"]["name"][0])
    ).iloc[:, cl_slice[0]:cl_slice[1]]

    dataset = ContrastiveCurveDataset(df_curves_cl, params_csv_path, augment_fn)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(42)
    train_dataset_cl, val_dataset_cl = random_split(
        dataset, [train_size, val_size], generator=split_generator
    )
    train_loader_cl = DataLoader(
        train_dataset_cl, batch_size=config["data"]["CL"]["batch_size"], shuffle=True
    )
    val_loader_cl = DataLoader(
        val_dataset_cl, batch_size=config["data"]["CL"]["batch_size"], shuffle=False
    )
    logging.info(
        "Prepared contrastive learning dataset with %s samples (%s train / %s val)",
        len(dataset),
        len(train_dataset_cl),
        len(val_dataset_cl),
    )

    learner = BYOL(
        transformer_model,
        image_size=image_size,
        hidden_layer="avgpool",
        config=config,
        F_config=f_config,
    ).to(device)

    logging.info("Starting contrastive learning training...")
    opt = torch.optim.AdamW(
        learner.parameters(),
        lr=config["cl_optimizer"]["lr"],
        weight_decay=config["cl_optimizer"]["weight_decay"],
        betas=config["cl_optimizer"].get("betas", (0.9, 0.999)),
    )
    total_steps = max(1, config["epochs_CL"] * len(train_loader_cl))
    warmup_steps = config["cl_optimizer"].get("warmup_steps", 0)
    scheduler = build_warmup_cosine_scheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_scale=config["cl_optimizer"].get("min_lr_scale", 0.01),
    )
    logging.info(
        "Using BYOL scheduler: warmup_steps=%s total_steps=%s min_lr_scale=%.4f",
        warmup_steps,
        total_steps,
        config["cl_optimizer"].get("min_lr_scale", 0.01),
    )
    pretrained_model = learner.online_encoder.net
    global_step = 0

    for epoch in range(config["epochs_CL"]):
        avg_loss = 0.0
        learner.train()
        epoch_num = epoch + 1
        train_bar = tqdm(
            train_loader_cl,
            desc=f"BYOL train {epoch_num}/{config['epochs_CL']}",
            leave=False,
            dynamic_ncols=True,
        )
        for i, (x, y) in enumerate(train_bar):
            global_step += 1
            x, y = x.to(device), y.to(device)
            loss = learner(x, y)
            avg_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            learner.update_moving_average()
            maybe_log_wandb(
                config,
                {
                    "byol/iteration_loss": loss.item(),
                    "byol/lr": get_current_lr(opt),
                    "byol/epoch": epoch + 1,
                },
                step=global_step,
            )
            train_bar.set_postfix(
                loss=f"{loss.item():.6f}",
                avg=f"{avg_loss / (i + 1):.6f}",
                lr=f"{get_current_lr(opt):.2e}",
            )

        avg_loss /= (i + 1)
        learner.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader_cl,
                desc=f"BYOL val {epoch_num}/{config['epochs_CL']}",
                leave=False,
                dynamic_ncols=True,
            )
            for val_i, (x_val, y_val) in enumerate(val_bar):
                x_val, y_val = x_val.to(device), y_val.to(device)
                batch_val_loss = learner(x_val, y_val)
                val_loss += batch_val_loss.item()
                val_bar.set_postfix(
                    loss=f"{batch_val_loss.item():.6f}",
                    avg=f"{val_loss / (val_i + 1):.6f}",
                )
        val_loss /= max(1, len(val_loader_cl))

        epoch_message = (
            f"BYOL Epoch {epoch_num}/{config['epochs_CL']} | "
            f"train_loss={avg_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={get_current_lr(opt):.8f} | "
            f"steps={global_step}"
        )
        print(epoch_message, flush=True)
        logging.info(epoch_message)
        maybe_log_wandb(
            config,
            {
                "byol/epoch_loss": avg_loss,
                "byol/val_loss": val_loss,
                "byol/lr": get_current_lr(opt),
                "byol/global_step": global_step,
            },
            step=epoch_num,
        )

        pretrained_model = learner.online_encoder.net

        if epoch_num % 20 == 0:
            model_path = os.path.join(run_context.run_output_dir, f"pretrained_model_CL_epoch_{epoch_num}.pth")
            save_checkpoint(
                pretrained_model,
                model_path,
                config,
                run_context,
                model_name="byol_contrastive",
                best_metric={"train_loss": float(avg_loss), "val_loss": float(val_loss)},
                step=epoch_num,
            )
            logging.info(
                "BYOL Epoch %s: Periodic checkpoint saved to %s (train_loss=%.6f, val_loss=%.6f)",
                epoch_num,
                model_path,
                avg_loss,
                val_loss,
            )

    return pretrained_model


def post_train_BYOL_CDAN(config, transformer_model, data_loaders, device, run_context, f_config, image_size, pretrained_cl_checkpoint):
    from byol_pytorch import BYOL
    from lars import LARS

    logging.info("Preparing BYOL+CDAN training...")
    learner = BYOL(
        transformer_model,
        image_size=image_size,
        hidden_layer="avgpool",
        config=config,
        F_config=f_config,
    ).to(device)

    logging.info("Initializing BYOL optimizer...")
    lars_optimizer = LARS(learner.parameters(), lr=0.01, momentum=0.9, weight_decay=1.5e-6)
    maybe_log_wandb(
        config,
        {"byol/init_lars_lr": get_current_lr(lars_optimizer)},
        step=0,
    )

    pretrained_model = learner.online_encoder.net
    load_matching_state_dict(pretrained_model, pretrained_cl_checkpoint, device)

    cdan_model = pretrained_model.to(device).train(True)
    random_layer, ad_net = build_domain_adversary(cdan_model, config, device)
    return train_CDAN(
        config,
        cdan_model,
        ad_net,
        random_layer,
        data_loaders,
        device,
        run_context,
        checkpoint_key="byol_cdan",
        model_name="byol_cdan",
    )


def finalize_run(config, run_context, final_results):
    save_json(os.path.join(run_context.run_output_dir, "final_results.json"), final_results)
    if is_wandb_enabled(config):
        wandb.summary.update(final_results)
        wandb.save(os.path.join(run_context.run_output_dir, "final_results.json"), base_path=run_context.run_output_dir)
        wandb.finish()
