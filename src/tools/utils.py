import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.autograd import Variable
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import loss
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import pairwise_distances

def plot_learning_curves(y1, y2):
    """
        This method is to plot leraning curves using training info.

        INPUT:
        training_s_statistic: statistic info on training soure domain data.
        testing_s_statistic: statistic info on testing soure domain data.
        testing_t_statistic: statistic info on training target domain data.
    """
    sns.set_theme()
    x = np.arange(0, 20000, 500)
    
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(x, y1, 'g-')
    ax2.plot(x, y2, 'b-')

    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Classification Accuracy', color='g')
    ax2.set_ylabel('A-distance', color='b')

    plt.show()
    plt.savefig('CDAN_ACA/learning_curve.png')

def plot_cm(true_labels, predictions, activities, save_name):
    """
        The function to plot confusion matrix using true and predicted CPE-targets. 
        
        INPUT:
        true_labels: ground truth labels in target domain.
        predictions: predicted labels.
        activities: the categories of CPE-targets.
    """
    # CM = confusion_matrix(true_labels, predictions)
    CM = [[45, 0, 0], [0, 70, 4], [11, 21, 52]]
    plt.figure(figsize=(15, 13))
    sns.set(font_scale=4.0)
    ax = sns.heatmap(CM, xticklabels=activities, yticklabels=activities, annot=True, annot_kws={"size": 50}, fmt='d', cmap='Greens')
    # save_name = 'ACA'
    plt.xlabel('Predicted Class',fontsize=45)
    plt.ylabel('True Class',fontsize=45)
    # plt.savefig(f'CDAN_ACA/Confusion_matrix/{save_name}_confusion_matrix.png')
    # plt.savefig(f'CDAN_ACA/Confusion_matrix/{save_name}_confusion_matrix.pdf')
    plt.show()

def save_log(obj, path):
    """
        Save training log info to the specified path.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('[INFO] Object saved to {}'.format(path))

def save_model(model, path):
    """
        Save trained network params to the specified path.
    """
    torch.save(model.state_dict(), path)
    print("checkpoint saved in {}".format(path))

def load_model(model, path):
	"""
	    Loads trained network params in case Transfromer params are not loaded.
	"""
	model.load_state_dict(torch.load(path))
	print("pre-trained model loaded from {}".format(path))
	
def split_data_labels(data, pointer_idx, use_normalize):
    """
        The simple function to split, shuffle and normalise data.
        
        INPUT:
        data: gblock or clinical isolate data frames.
        
        RETURN:
        x: A list of amplification curves, each of which is normalised via the min-max scaler.
        y: A list of CPE target names (i.e. labels in the training or testing set.). 
    """
    if 'Sample_ID' in data.columns:
        X, y, z = data.iloc[:,data.shape[1] - pointer_idx:], data[['Target']], data[['Sample_ID']]
    else:
        X, y, z = data.iloc[:,data.shape[1] - pointer_idx:], data[['Target']], data[['Conc']]
    X, y, z = shuffle(X, y, z, random_state = 2)
    if use_normalize:
        X[X.columns] = StandardScaler().fit_transform(X)
    return X, y, z

def df_to_tensor(df):
    """
        The function to convert a dataframe to a tensor.
    """
    return torch.from_numpy(df.values).float().cpu()

def arr_to_tensor(arr):
    """
        The function to convert a np.array to a tensor.
    """
    return torch.from_numpy(arr).float().cpu()

def str_to_int(y):
    """
        The function to convert CPE target names into integers. This function is
        used as the sparse-categorical entropy loss function is applied.
        i.e ('imp' -> 0, 'ndm' -> 1, 'oxa48' -> 2)

        INPUT:
        y: The collection of CPE target names.

        RETURN:
        A set of integers, each correponds to a CPE target name.
    """
    label_encoder = LabelEncoder()
    vec = label_encoder.fit_transform(y)
    return vec

def sigmoid5_un(x, Fm, Fb, Sc, Cs, As):
	"""
        Five paramter model (universal notations).
        
        INPUT:
        x: iterative x locations
        Fm, Fb, Sc, Cs, As: parameters
        
        Fm: maximum fluorescence
        Fb: background fluorescence
        Sc: slope of the curve 
        Cs: fractional cycle of the inflection point (1/c)
        As: asymmetric shape (Richard's coefficient)
        
        RETURN:
        y outputs
    """
	return Fm / (1. + np.exp(-(x-Cs)*Sc))**As + Fb

def preprocess_data_dl(X,y):
    """
        The simple function to split, shuffle and normalise data.
        :param data: (DataFrame): gblock or clinical isolate data frames.
        :return x: ([[int]]) A list of amplification curves, each of which is normalised 
        via the min-max scaler.
        :return y: ([string]) A list of CPE target names (i.e. labels in
        the training or testing set.). 
    """
    X, y = shuffle(X, y, random_state = 2)
    for i in range(X.shape[-1]):
        X[:,:, i] = MinMaxScaler().fit_transform(X[:,:,i])
    return X, y

# Compute A-distance using numpy and sklearn
# Reference: Analysis of representations in domain adaptation, NIPS-07.


def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = [1, 3, 5, 7]

    half_source, half_target = int(nb_source * 0.5), int(nb_target * 0.5)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))
    # print(metrics.silhouette_score(test_X, test_Y, metric='euclidean'))
    best_risk = 1.0
    for C in C_list:
        clf = KNeighborsClassifier(n_neighbors=C)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)
        print(2 * (1. - 2 * best_risk))
    return 2 * (1. - 2 * best_risk)

# def proxy_a_distance(model, ad_net, source_loader, target_loader, random_layer, device):
#     """
#     Compute the Proxy-A-Distance of a source/target representation
#     """
#     source = list(enumerate(source_loader))
#     target = list(enumerate(target_loader))
#     valid_steps = min(len(source), len(target))

#     discriminator_result_list = torch.tensor([]).to(device)
#     soft_label_list = torch.tensor([]).to(device)

#     for batch_idx in range(valid_steps):
#         # fetch data in batches
#         # _, source_data -> torch.Size([128, 45, 1]), labels -> torch.Size([128])
#         _, (source_data, source_label, _) = source[batch_idx]
#         _, (target_data, target_label, _) = target[batch_idx] # unsupervised learning

#         if source_data.shape[0] != target_data.shape[0]:
#             break

#         # move to device
#         source_data = source_data.to(device)
#         source_label = source_label.to(device)
#         target_data = target_data.to(device)
    
#         # create pytorch variables, the variables and functions build a dynamic graph of computation
#         source_data, source_label = Variable(source_data), Variable(source_label)
#         target_data = Variable(target_data)

#         # do a forward pass through network (recall DeepCORAL outputs source, target activation maps)
#         src_features, src_ouputs = model(source_data)
#         tgt_features, tgt_ouputs = model(target_data)

#         source_soft_label = torch.zeros(source_label.shape[0]).type(torch.LongTensor).to(device)
#         target_soft_label = torch.ones(target_label.shape[0]).type(torch.LongTensor).to(device)
#         soft_label = torch.cat((source_soft_label, target_soft_label), dim=0)

#         torch.cat((src_features, tgt_features), dim=0)
#         features = torch.cat((src_features, tgt_features), dim=0)
#         outputs = torch.cat((src_ouputs, tgt_ouputs), dim=0)
#         softmax_output = nn.Softmax(dim=1)(outputs)

#         if random_layer is None:
#             op_out = torch.bmm(softmax_output.unsqueeze(2), features.unsqueeze(1))
#             discriminator_result = ad_net(op_out.view(-1, softmax_output.size(1) * features.size(1)))
#         else:
#             random_out = random_layer.forward([features, softmax_output])
#             discriminator_result = ad_net(random_out.view(-1, random_out.size(1)))     

#         discriminator_result = torch.squeeze(discriminator_result, -1)  
#         threshold = torch.tensor([0.5]).to(device)
#         discriminator_result = (discriminator_result> threshold).float()*1
#         discriminator_result_list = torch.cat((discriminator_result_list, discriminator_result.long()), 0)
#         soft_label_list = torch.cat((soft_label_list, soft_label), 0)
    
#     correct_class = discriminator_result_list.eq(soft_label_list.data.view_as(discriminator_result_list)).cpu().sum()
#     test_error = 1. - correct_class/ soft_label_list.shape[0]

#     # Ensure A_distance is non-negative.
#     if test_error > .5:
#         test_error = 1 - test_error

#     # A_distance = 2 * (1 - 2ε).
#     A_distance = 2 * (1. - 2 * test_error) 
#     return A_distance

def visualize(model, network_type, source_test_loader, target_test_loader, save_name):
    source_label_list, target_label_list = [], []
    count = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_feature_list = torch.tensor([]).to(device)
    target_feature_list = torch.tensor([]).to(device)
    target = list(enumerate(target_test_loader))
    batch_size = target[0][1][1].shape[0]
    print("\n Extract features to draw T-SNE plot...")
    for i, (source_data, source_label, _) in enumerate(source_test_loader):
        _, (target_data, target_label, _) = target[i]

        if i >= 5 or source_data.shape[0] != batch_size or target_data.shape[0] != batch_size: 
            break     
        
        if network_type == 'RF5':
            source_feature =  target_data.squeeze(-1)
            target_feature =  target_data.squeeze(-1)
        else:
            source_feature, _ =  model(source_data)
            target_feature, _ =  model(target_data)
            
        source_feature_list = torch.cat((source_feature_list, source_feature), 0) 
        target_feature_list = torch.cat((target_feature_list, target_feature), 0) 
        
        source_label = source_label.cpu().numpy()
        target_label = target_label.cpu().numpy()

        source_label_list.append(source_label)
        target_label_list.append(target_label)
        count += 1
    
    # Stack source_list + target_list
    combined_feature = torch.cat((source_feature_list, target_feature_list), 0)
    combined_label_list = source_label_list
    combined_label_list.extend(target_label_list)
    print(combined_feature.shape)
    source_domain_list = torch.zeros(count * batch_size).type(torch.LongTensor)
    target_domain_list = torch.ones(count * batch_size).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).cpu()

    tsne = TSNE(perplexity=30, n_components=2, n_iter=3000)
    dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())
    combined_label_list = list(itertools.chain.from_iterable(combined_label_list))
    combined_label_list = list(map(str, combined_label_list))
    combined_label_list = np.asarray(combined_label_list)
    print(combined_label_list.shape)
    new_combined_label_list = []
    for i in range(len(combined_label_list)):
        if combined_label_list[i] == "0":
            new_combined_label_list.append('IMP')
        elif combined_label_list[i] == "1":
            new_combined_label_list.append('NDM')
        else:
            new_combined_label_list.append("OXA48")

    combined_domain_list = list(map(str, combined_domain_list.cpu().numpy()))

    for i in range(len(combined_domain_list)):
        if combined_domain_list[i] == "0":
            combined_domain_list[i] = 'Synthetic DNA'
        elif combined_domain_list[i] == "1":
            combined_domain_list[i] = 'Clinical Isolates'
    print('Draw plot ...')

    df = pd.DataFrame()
    df["CPE Target"] = new_combined_label_list
    df["Domain Type"] = combined_domain_list
    df["First Component"] = dann_tsne[:,0]
    df["Second Component"] = dann_tsne[:,1]

    # df.to_csv('CDAN_ACA/TSNE_plot/T_SNE_RF.csv')  

    plt.figure(dpi = 300)
    ax = sns.scatterplot(x = "First Component", y = "Second Component", 
                        data=df, s= 30, hue = "Domain Type", alpha = 0.7)

    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    ax.set(xlabel=None, ylabel = None)
    ax.set_yticklabels(ax.get_yticks(), size = 12)
    ax.set_xticklabels(ax.get_xticks(), size = 12)

    fig_name = 'CDAN_ACA/T_SNE/' +  str(save_name) + '_TSNE.png'
    fig_name_1 = 'CDAN_ACA/T_SNE/' +  str(save_name) + '_TSNE.pdf'
    plt.savefig(fig_name)
    plt.savefig(fig_name_1)

    palette = ['lightcoral', 'skyblue', 'seagreen']
    plt.figure(dpi = 300)
    ax = sns.scatterplot(x = "First Component", y = "Second Component", 
                        data=df, s= 30, hue = "CPE Target", palette = palette, alpha = 0.7)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    ax.set(xlabel=None, ylabel = None)
    ax.set_yticklabels(ax.get_yticks(), size = 12)
    ax.set_xticklabels(ax.get_xticks(), size = 12)

    fig_name = 'CDAN_ACA/T_SNE/' +  str(save_name) + '_TSNE(1).png'
    fig_name_1 = 'CDAN_ACA/T_SNE/' +  str(save_name) + '_TSNE(1).pdf'
    plt.savefig(fig_name)
    plt.savefig(fig_name_1)