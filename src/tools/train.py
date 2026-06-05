
import loss
import torch
import torch.nn as nn
import lr_schedule
# from test import test
import network
import sys
from utils import visualize
import os
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for headless environments
import matplotlib.pyplot as plt


def test(model, data_loader, iter, device):
    """
        This method is used for evaluating the performance of model on the testing 
        target domain data.

        Implementation based on:
        https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py

        INPUT:
        model: Generator network.
        data_loader: target domain dataloader.
        iter: the current iteration number for testing.
        device: choose cuda/cpu device to train models.

        RETURN:
        results: A dictionary containing test info of at specific.
    """
    model.eval()
    test_loss = 0
    correct_class = 0

    # go over dataloader batches, labels
    for data, label in data_loader:
        data, label = data.to(device), label.to(device)
        # note on volatile: https://stackoverflow.com/questions/49837638/what-is-volatile-variable-in-pytorch
        data, label = Variable(data, volatile=True), Variable(label)
        _, output = model(data) # just use one ouput of DeepCORAL

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct_class += pred.eq(label.data.view_as(pred)).cpu().sum()

    # compute test loss as correclty classified labels divided by total data size
    test_loss = test_loss/len(data_loader.dataset)

    # return dictionary containing info of each epoch
    return {
        "iter": iter,
        "average_loss": test_loss,
        "correct_class": correct_class,
        "total_elems": len(data_loader.dataset),
        "accuracy %": 100.*correct_class/len(data_loader.dataset)
    }

def train(config, base_network, ad_net, random_layer, data_loaders, device):
    """
        This method feeds the input data to the generator, label predictor, and domain classifier model 
        and start training.

        Implementation based on:
        https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py

        INPUT:
        config: framework configuration, including optimizer for training, evaluate model 
        performance after how many steps, etc.
        ad_net: Domain classifier network.
        random_layer: Random projection layer for calculating A_distance.
        dset_loader: A dictionary which stores source domain, target domain and testing dataloaders.
        device: choose cuda/cpu device to train models.

        RETURN:
        A_distance_list, Acc_list
    """
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
                    
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train   
    len_train_source = len(data_loaders["source"])
    len_train_target = len(data_loaders["target"])
    best_acc = 0.0
    best_model = None

    base_network.train(True)
    for i in range(config["num_iterations"]):

        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            test_target = test(base_network, data_loaders["source"], i, device)
            temp_acc = test_target['accuracy %']
            temp_model = nn.Sequential(base_network)
            # visualize(base_network, 'Transformer', dset_loaders["source"], dset_loaders["target"], 'Transformer')
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
                # torch.save(best_model, '{0}/TST_ACA_DA_Generator.pth'.format('CDAN_ACA/model')) 
                # torch.save(ad_net, '{0}/TST_ACA_DA_Discriminator.pth'.format('CDAN_ACA/model'))
            log_str = "[iter: {:04d} / all {:05d}], Accuracy: {:.5f}".format(i+1, config["num_iterations"], temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(f'\n{log_str}') # print testing info.

        # train one iter.
        # base_network.train(True)
        # ad_net.train(True)

        loss_params = config["loss"]  
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(data_loaders["source"])

        # if i % len_train_target == 0:
        #     iter_target = iter(data_loaders["target"])
        
        inputs_source, labels_source = next(iter_source)

        # inputs_target, _ = next(iter_target)
        # inputs_source, inputs_target, labels_source = inputs_source.to(device), inputs_target.to(device), labels_source.to(device)
        
        inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)
        features_source, outputs_source = base_network(inputs_source)
        # features_target, outputs_target = base_network(inputs_target)
        # features = torch.cat((features_source, features_target), dim=0)
        # outputs = torch.cat((outputs_source, outputs_target), dim=0)
        # softmax_out = nn.Softmax(dim=1)(outputs)

        # if config['method'] == 'CDAN+E':           
        #     entropy = loss.Entropy(softmax_out)
        #     _, transfer_loss = loss.CDAN([features, softmax_out], ad_net, device, entropy, network.calc_coeff(i), random_layer)
        # elif config['method']  == 'CDAN':
        #     _, transfer_loss = loss.CDAN([features, softmax_out], ad_net, device, None, None, random_layer)
        # elif config['method']  == 'DANN':
        #     _, transfer_loss = loss.DANN(features, ad_net)
        # else:
        #     raise ValueError('Method cannot be recognized.')

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        # total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss = classifier_loss
        total_loss.backward()
        optimizer.step()

        # print('Classification loss: %f, Total_Loss: %f' % (classifier_loss.item(), total_loss.item()))
        # print training info.
        sys.stdout.write('\r[iter: %d / all %d], Classification loss: %f, CDAN loss: %f, Total_Loss: %f, best_acc: %f' \
              % (i+1, config["num_iterations"], classifier_loss.item(), 0, total_loss.item(), best_acc))
        sys.stdout.flush()

    # torch.save(best_model, '{0}/TST_ACA_DA_Generator.pth'.format('CDAN_ACA/model')) 
    # torch.save(ad_net, '{0}/TST_ACA_DA_Discriminator.pth'.format('CDAN_ACA/model')) 
    best_model = nn.Sequential(base_network)
    
    return best_acc, best_model

def train_2(config, base_network, ad_net, random_layer, data_loaders, device):
    
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    dir_out = config["output_path"]

    len_train_source = len(data_loaders["source"])
    best_acc = 0.0
    best_model = None

    # Initialize lists to store training and testing accuracy
    training_accuracy = []
    testing_accuracy = []

    base_network.train(True)
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            
            # Evaluate on test data
            test_target = test(base_network, data_loaders["test"], i, device)
            temp_acc = test_target['accuracy %']
            temp_model = nn.Sequential(base_network)
            
            # Save best model if this iteration has better accuracy
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model

            # Log testing accuracy
            testing_accuracy.append((i + 1, temp_acc))
            
            # Print testing info
            log_str = "[iter: {:04d} / all {:05d}], Testing Accuracy: {:.5f}".format(i + 1, config["num_iterations"], temp_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(f'\n{log_str}')

        # Train one iteration
        loss_params = config["loss"]
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(data_loaders["source"])

        inputs_source, labels_source = next(iter_source)
        inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)
        features_source, outputs_source = base_network(inputs_source)

        # Calculate classification loss
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        total_loss = classifier_loss
        total_loss.backward()
        optimizer.step()

        # Compute training accuracy for the current batch
        preds = outputs_source.argmax(dim=1)
        train_acc = (preds == labels_source).float().mean().item()
        training_accuracy.append((i + 1, train_acc))

        # Print training info
        sys.stdout.write('\r[iter: %d / all %d], Classification loss: %f, Training Accuracy: %f, Testing Accuracy: %f, Best Testing Accuracy: %f' %
                         (i + 1, config["num_iterations"], classifier_loss.item(), train_acc, temp_acc if i % config["test_interval"] == config["test_interval"] - 1 else 0, best_acc))
        sys.stdout.flush()

    # Save best model
    best_model = nn.Sequential(base_network)

    # Save training and testing curves
    training_curve = {"training_accuracy": training_accuracy, "testing_accuracy": testing_accuracy}
    # Extract training and testing accuracy data
    train_iters, train_acc = zip(*training_curve["training_accuracy"])
    test_iters, test_acc = zip(*training_curve["testing_accuracy"])

    # Convert training accuracy to percentages
    train_acc_percentage = [x * 100 for x in train_acc]

    # Plot training and testing accuracy curves
    plt.figure()
    plt.plot(train_iters, train_acc_percentage, label="Training Accuracy (%)")
    plt.plot(test_iters, test_acc, label="Testing Accuracy (%)")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy Curves")
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], "training_curve.png"))  # Save plot to file

    # Save the training curve to dir_out``
    output_path = os.path.join(config["output_path"], "training_curve.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(training_curve, f)
    print(f"Training curve saved to {output_path}")

    return best_acc, best_model, training_curve

def train_CDAN(config, base_network, ad_net, random_layer, data_loaders, device):
    """
        This method feeds the input data to the generator, label predictor, and domain classfier model 
        and start training.

        Implementation based on:
        https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py

        INPUT:
        config: framework configuration, including optimizer for training, evaluate model 
        performance after how many steps, etc.
        ad_net: Domain classifier network.
        random_layer: Random projection layer for calculating A_distance.
        dset_loader: A dictionary which stores source domain, target domain and testing dataloaders.
        device: choose cuda/cpu device to train models.

        RETURN:
        A_distance_list, Acc_list
    """
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
                    
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train   
    len_train_source = len(data_loaders["source"])
    len_train_target = len(data_loaders["target"])
    best_acc = 0.0
    best_model = None
    # Initialize lists to store training and testing accuracy
    training_accuracy = []
    testing_accuracy = []

    base_network.train(True)

    classifier_loss_iter = []
    transfer_loss_iter = []
    iters = 0

    for i in range(config["num_iterations"]):

        iters += 1
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            
            # Evaluate on test data
            test_target = test(base_network, data_loaders["test"], i, device)
            temp_acc = test_target['accuracy %']
            temp_model = nn.Sequential(base_network)
            
            # Save best model if this iteration has better accuracy
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model

            # Log testing accuracy
            testing_accuracy.append((i + 1, temp_acc))
            
            # Print testing info
            log_str = "[iter: {:04d} / all {:05d}], Testing Accuracy: {:.5f}".format(i + 1, config["num_iterations"], temp_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(f'\n{log_str}')

        # train one iter.
        # base_network.train(True)
        # ad_net.train(True)

        loss_params = config["loss"]  
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(data_loaders["source"])

        if i % len_train_target == 0:
            iter_target = iter(data_loaders["target"])
        
        inputs_source, labels_source = next(iter_source)
        inputs_target, _= next(iter_target)
        inputs_source, inputs_target, labels_source = inputs_source.to(device), inputs_target.to(device), labels_source.to(device)
        
        # inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        if config['method'] == 'CDAN+E':           
            entropy = loss.Entropy(softmax_out)
            _, transfer_loss = loss.CDAN([features, softmax_out], ad_net, device, entropy, network.calc_coeff(i), random_layer)
        elif config['method']  == 'CDAN':
            _, transfer_loss = loss.CDAN([features, softmax_out], ad_net, device, None, None, random_layer)
        elif config['method']  == 'DANN':
            _, transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        # total_loss = classifier_loss
        total_loss.backward()
        optimizer.step()

        classifier_loss_iter.append(classifier_loss.item())
        transfer_loss_iter.append(transfer_loss.item())

        # print('Classification loss: %f, Total_Loss: %f' % (classifier_loss.item(), total_loss.item()))
        # print training info.
        sys.stdout.write('\r[iter: %d / all %d], Classification loss: %f, CDAN loss: %f, Total_Loss: %f' \
              % (i+1, config["num_iterations"], classifier_loss.item(), transfer_loss.item(), total_loss.item()))
        sys.stdout.flush()
    
    # Plot training and testing accuracy curves
    plt.figure()
    plt.plot(range(iters), classifier_loss_iter, label="classifier_loss_iter")
    plt.plot(range(iters), transfer_loss_iter, label="transfer_loss_iter")
    plt.xlabel("Iterations")
    plt.ylabel("Loss functions")
    plt.title("classifier and CDAN losses")
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], "training_curve_CDAN.png"))  # Save plot to file

    # torch.save(best_model, '{0}/TST_ACA_DA_Generator.pth'.format('CDAN_ACA/model')) 
    # torch.save(ad_net, '{0}/TST_ACA_DA_Discriminator.pth'.format('CDAN_ACA/model')) 
    # best_model = nn.Sequential(base_network)
    best_model = base_network
    
    return best_acc, best_model