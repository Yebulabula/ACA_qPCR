import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'byol-pytorch'))

import argparse
import logging
import warnings

import numpy as np
import torch
import torch.optim as optim
from basenetwork import Transformer_for_byol
from tools.data_loader import generate_data_loader

from byol_cdan_common import (
    build_domain_adversary,
    configure_logging,
    create_run_context,
    eval_best_model,
    finalize_run,
    load_matching_state_dict,
    maybe_init_wandb,
    post_train_BYOL_CDAN,
    save_checkpoint,
    save_json,
    train_BYOL,
    train_CDAN,
    train_baseline,
)


dir_data_2025 = os.path.join(BASE_DIR, '7_plex_data')
dir_out = os.path.join(BASE_DIR, '7_plex_output')
PRETRAINED_BYOL_CHECKPOINT = os.path.join(BASE_DIR, "output", "4.3.7_plex_best_byol_qPCR.pth")
PRETRAINED_CL_CHECKPOINT = os.path.join(dir_out, "pretrained_model_CL_final.pth")
PARAMS_CSV_PATH = os.path.join(dir_data_2025, "param_df_5_20250305_2248.csv")


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def sigmoid5_un(x, Fm, Fb, Sc, Cs, As):
    return Fm / (1.0 + np.exp(-(x - Cs) * Sc)) ** As + Fb


def augment_fct(params_df, idx, sigmoid_fn=sigmoid5_un):
    param_cols = ['Fm', 'Fb', 'Sc', 'Cs', 'As']
    params = params_df.loc[idx, param_cols]
    params_aug = params.copy()

    mean = 24.835403682791412
    sigma = 8.098645591811588
    dist = np.random.normal(loc=mean, scale=sigma, size=len(params_df))
    params_aug['Cs'] = np.random.choice(dist)
    x = np.arange(1, 60 + 1)
    return sigmoid_fn(x, *params_aug)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--dir_data', type=str, default=dir_data_2025, help="directory of data")
    parser.add_argument('--dir_out', type=str, default=dir_out, help="output directory of our model")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size for source/target/test training loaders")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="optimizer weight decay")
    parser.add_argument('--trade_off', type=float, default=1.0, help="domain adaptation trade-off weight")
    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'adamw'], help="optimizer for baseline/CDAN training")
    parser.add_argument('--train_warmup_steps', type=int, default=0, help="warmup steps for baseline/CDAN scheduler")
    parser.add_argument('--train_min_lr_scale', type=float, default=0.05, help="minimum lr scale for baseline/CDAN scheduler")
    parser.add_argument('--random', type=str2bool, default=False, help="whether use random projection")
    parser.add_argument('--run_cl', type=str2bool, default=False, help="whether to run BYOL contrastive learning pretraining only")
    parser.add_argument('--run_baseline', type=str2bool, default=False, help="whether to run baseline model training")
    parser.add_argument('--run_cdan', type=str2bool, default=False, help="whether to run CDAN model training")
    parser.add_argument('--run_byol_cdan', type=str2bool, default=False, help="whether to run BYOL+CDAN model training")
    parser.add_argument('--wandb', action='store_true', help="enable Weights & Biases logging")
    parser.add_argument('--wandb_project', type=str, default='byol-cdan-7-plex', help="wandb project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="wandb entity/team name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="optional wandb run name")
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'], help="wandb mode")
    parser.add_argument('--pretrained_cl_checkpoint', type=str, default=PRETRAINED_CL_CHECKPOINT, help="path to the BYOL contrastive checkpoint used to initialize T-CDAN")
    parser.add_argument('--eval_checkpoint', type=str, default=None, help="optional classifier checkpoint to evaluate without training")
    parser.add_argument('--eval_split', type=str, default='test', choices=['source', 'target', 'test'], help="dataset split to use for checkpoint-only evaluation")
    args, _ = parser.parse_known_args()

    run_context = create_run_context(args.dir_out, use_timestamp_subdir=False, write_log_file=False)
    configure_logging(run_context)
    logging.info("Arguments: %s", args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Device for training: %s', device)
    optimizer_type = optim.Adam if args.optimizer_type == 'adam' else optim.AdamW

    config = {
        "method": args.method,
        "N_seq": 60,
        "num_iterations": args.num_iterations,
        "epochs_CL": 100,
        "test_interval": args.test_interval,
        "dir_out": args.dir_out,
        "loss": {"trade_off": args.trade_off, "random": args.random, "random_dim": 512},
        "optimizer": {"type": optimizer_type, "optim_params": {'lr': args.lr, "weight_decay": args.weight_decay}, "lr_type": "inv",
                      "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}},
        "train_scheduler": {
            "warmup_steps": args.train_warmup_steps,
            "min_lr_scale": args.train_min_lr_scale,
        },
        "cl_optimizer": {
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "betas": (0.9, 0.999),
            "warmup_steps": 100,
            "min_lr_scale": 0.02,
        },
        "data": {"dir_data": args.dir_data,
                 "source": {"name": ['df_dPCR_GB_2025.csv'], "batch_size": args.batch_size},
                 "target": {"name": "df_dPCR_GB_2025.csv", "batch_size": args.batch_size},
                 "test": {"name": "df_qPCR_GB_2025.csv", "batch_size": args.batch_size},
                 "CL": {"name": ["df_dPCR_SP_2025.csv"], "batch_size": 1024},
                 "normalize": "min_max"},
        "class_num": 7,
        "wandb": {
            "enabled": bool(args.wandb and args.wandb_mode != 'disabled'),
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name,
            "mode": args.wandb_mode,
            "tags": [args.method.lower(), "7-plex"],
        },
        "checkpoint_names": {
            "baseline": "baseline.pth",
            "cdan": "cdan.pth",
            "byol_cdan": "byol_cdan.pth",
        },
    }

    F_config = {
        'd_input': 1,
        'd_model': 128,
        'q': 8,
        'v': 8,
        'h': 4,
        'N': 4,
        'attention_size': None,
        'dropout': 0.3,
        'chunk_mode': None,
        'pe': "regular",
        'pe_period': 20,
        'use_bottleneck': True,
        "bottleneck_dim": 256
    }

    logging.info("CDAN Configuration: %s", config)
    logging.info("Transformer Configuration: %s", F_config)
    torch.manual_seed(42)
    maybe_init_wandb(
        config,
        args,
        F_config,
        run_context,
        PRETRAINED_BYOL_CHECKPOINT,
        args.pretrained_cl_checkpoint,
        "byol-cdan-7-plex",
    )

    logging.info("Loading data...")
    data_config = config["data"]
    data_loaders = generate_data_loader(
        data_config["dir_data"],
        data_config["source"]["batch_size"],
        data_config["target"]["batch_size"],
        data_config["test"]["batch_size"],
        data_config["source"]["name"],
        data_config["target"]["name"],
        data_config["test"]["name"],
        device,
        use_normalize='None',
    )
    logging.info("Data loading complete!")

    activities = ['Adeno', 'COVID', 'Cov_229E', 'Cov_HKU1', 'Cov_NL63', 'Cov_OC43', 'MERS']
    final_results = {}

    if args.eval_checkpoint:
        logging.info("Running checkpoint-only evaluation from: %s", args.eval_checkpoint)
        eval_model = Transformer_for_byol(config['class_num'], config['N_seq'], **F_config).to(device)
        load_matching_state_dict(eval_model, args.eval_checkpoint, device, strip_prefixes=True)
        eval_metrics = eval_best_model(
            eval_model,
            data_loaders,
            activities,
            'CheckpointEval',
            device,
            run_context,
            args.eval_split,
            config=config,
        )
        final_results[f'checkpoint_eval_{args.eval_split}'] = eval_metrics["accuracy"]
        logging.info(
            "Checkpoint evaluation on %s split complete. Accuracy: %.2f%%",
            args.eval_split,
            eval_metrics["accuracy"],
        )
        finalize_run(config, run_context, final_results)
        sys.exit(0)

    if args.run_cl:
        logging.info("\n\n=========== STARTING BYOL CONTRASTIVE LEARNING ===========")
        byol_transformer = Transformer_for_byol(config['class_num'], config['N_seq'], **F_config).to(device)
        cl_pretrained_model = train_BYOL(
            config,
            byol_transformer,
            device,
            run_context,
            F_config,
            image_size=60,
            cl_slice=(3, 63),
            params_csv_path=PARAMS_CSV_PATH,
            augment_fn=augment_fct,
        )
        if cl_pretrained_model is not None:
            final_cl_path = os.path.join(run_context.run_output_dir, "pretrained_model_CL_final.pth")
            save_checkpoint(
                cl_pretrained_model,
                final_cl_path,
                config,
                run_context,
                model_name="byol_contrastive_final",
                step=config["epochs_CL"],
            )
            logging.info("Final BYOL contrastive model saved to: %s", final_cl_path)
            final_results["BYOL_CL"] = "completed"

    if args.run_baseline:
        logging.info("\n\n=========== STARTING BASELINE MODEL TRAINING ===========")
        baseline_transformer = Transformer_for_byol(config['class_num'], config['N_seq'], **F_config).to(device)
        baseline_best_model = train_baseline(config, baseline_transformer, data_loaders, device, run_context)
        baseline_metrics = eval_best_model(
            baseline_best_model, data_loaders, activities, 'Baseline', device, run_context, 'test', config=config
        )
        final_results['Baseline'] = baseline_metrics["accuracy"]
        logging.info("Baseline Model Final Accuracy: %.2f%%", baseline_metrics['accuracy'])

    if args.run_cdan:
        logging.info("\n\n=========== STARTING CDAN MODEL TRAINING ===========")
        cdan_transformer = Transformer_for_byol(config['class_num'], config['N_seq'], **F_config).to(device)
        random_layer, ad_net = build_domain_adversary(cdan_transformer, config, device)
        cdan_best_model = train_CDAN(config, cdan_transformer, ad_net, random_layer, data_loaders, device, run_context)
        cdan_metrics = eval_best_model(
            cdan_best_model, data_loaders, activities, 'CDAN', device, run_context, 'test', config=config
        )
        final_results['CDAN'] = cdan_metrics["accuracy"]
        logging.info("CDAN Model Final Accuracy: %.2f%%", cdan_metrics['accuracy'])

    if args.run_byol_cdan:
        byol_cdan_transformer = Transformer_for_byol(config['class_num'], config['N_seq'], **F_config).to(device)
        byol_cdan_best_model = post_train_BYOL_CDAN(
            config,
            byol_cdan_transformer,
            data_loaders,
            device,
            run_context,
            F_config,
            image_size=60,
            pretrained_cl_checkpoint=args.pretrained_cl_checkpoint,
        )
        byol_cdan_metrics = eval_best_model(
            byol_cdan_best_model, data_loaders, activities, 'BYOL_CDAN', device, run_context, 'test', config=config
        )
        final_results['BYOL+CDAN'] = byol_cdan_metrics["accuracy"]
        logging.info("BYOL+CDAN Model Final Accuracy: %.2f%%", byol_cdan_metrics['accuracy'])

    if len(final_results) > 1:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        numeric_results = {k: v for k, v in final_results.items() if isinstance(v, (int, float))}
        plt.bar(numeric_results.keys(), numeric_results.values())
        plt.xlabel('Method')
        plt.ylabel('Accuracy (%)')
        plt.title('Comparison of Different Training Methods')
        plt.ylim([0, 100])
        for i, v in enumerate(numeric_results.values()):
            plt.text(i, v + 1, f"{v:.2f}%", ha='center')
        compare_path = os.path.join(run_context.run_output_dir, "method_comparison.png")
        plt.savefig(compare_path)
        plt.close()
        logging.info("Comparison chart saved to: %s", compare_path)

    finalize_run(config, run_context, final_results)
    logging.info("All training and evaluation complete!")
