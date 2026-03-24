import loss
import torch
import torch.nn as nn
import lr_schedule
from test import test
import network
import sys
from utils import visualize

def train(config, base_network, data_loaders, device):
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
    parameter_list = base_network.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
                    
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train   
    len_train_source = len(data_loaders["source"])
    len_train_target = len(data_loaders["target"])
    best_acc = 0.0
    best_model = None

    for i in range(config["num_iterations"]):

        if i % config["test_interval"] == config["test_interval"] - 1:
            best_model = base_network
            print('save')
            torch.save(best_model, '{0}/TST_ACA_DA_Generator.pth'.format('CDAN_ACA/model')) 
            # base_network.train(False)
            # test_target = test(base_network, data_loaders["target"], i, device)
            # temp_acc = test_target['accuracy %']
            # temp_model = nn.Sequential(base_network)
            # # visualize(base_network, 'Transformer', dset_loaders["source"], dset_loaders["target"], 'Transformer')
            # if temp_acc > best_acc:
            #     best_acc = temp_acc
            #     best_model = temp_model
            #     torch.save(best_model, '{0}/TST_ACA_DA_Generator.pth'.format('CDAN_ACA/model')) 
            # log_str = "[iter: {:04d} / all {:05d}], Accuracy: {:.5f}".format(i+1, config["num_iterations"], temp_acc)
            # config["out_file"].write(log_str+"\n")
            # config["out_file"].flush()
            # print(f'\n{log_str}') # print testing info.

        # train one iter.
        base_network.train(True)

        loss_params = config["loss"]  
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(data_loaders["source"])

        if i % len_train_target == 0:
            iter_target = iter(data_loaders["target"])
        
        inputs_source, labels_source = iter_source.next()
        inputs_target, _= iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.to(device), inputs_target.to(device), labels_source.to(device)
        
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

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
        total_loss =  classifier_loss
        total_loss.backward()
        optimizer.step()

        # print training info.
        sys.stdout.write('\r[iter: %d / all %d], Classification loss: %f, Total_Loss: %f' \
              % (i+1, config["num_iterations"], classifier_loss.item(), total_loss.item()))
        sys.stdout.flush()

    torch.save(best_model, '{0}/TST_ACA_DA_Generator.pth'.format('CDAN_ACA/model')) 
    # torch.save(ad_net, '{0}/TST_ACA_DA_Discriminator.pth'.format('CDAN_ACA/model')) 
    # best_model = nn.Sequential(base_network)
    
    return best_acc, best_model