from sklearn.neighbors import KNeighborsClassifier
import torch
from base_GD import GD_Base_VAE
from GD_Gauss import GD
from DVAE_p1 import DVAE
from sparse_VAE import DirSparse_VAE
from podLDA import podLDA
import torchvision
import numpy as np
from torchvision import transforms
import os
import json

train_valid_test_splits = (45000, 5000, 10000)


seed = 1234
dataloader_kwargs = {}
CUDA = torch.cuda.is_available()
download_needed = not os.path.exists('./MNIST')
model_path = 'trained_models'
checkpoint_path = None

if CUDA:
    torch.cuda.manual_seed(seed)
    dataloader_kwargs.update({'num_workers': 1, 'pin_memory': True})

# get datasets
train_dataset = torchvision.datasets.MNIST('.', train=True, transform=transforms.ToTensor(), download=download_needed)
test_dataset = torchvision.datasets.MNIST('.', train=False, transform=transforms.ToTensor())

# get dimension info
input_shape = list(train_dataset.data[0].shape)
input_ndims = np.product(input_shape)

# define data loaders
train_data = train_dataset.data.reshape(-1, 1, *input_shape) / 255  # reshaping and scaling bytes to [0,1]
test_data = test_dataset.data.reshape(-1, 1, *input_shape) / 255
pruned_train_data = train_data[:train_valid_test_splits[0]]

model_names = ['G_D', 'GD_Base_VAE', 'DirSparse_VAE', 'p_o_d_L_D_A', 'RawPixels']
xy_sets = ['train_data', 'train_labels', 'test_data', 'test_labels']
k = [2, 3, 4, 5, 6, 7, 8, 9]
checkpoint_paths = ['/home/akinlolu/Desktop/newCode/Dirichlet-VAE-main/trained_models_MNIST/G_D/best_checkpoint_G_D_Jul_13_2023_13_21',
                    '/home/akinlolu/Desktop/newCode/Dirichlet-VAE-main/trained_models_MNIST/GD_Base_VAE/best_checkpoint_GD_Base_VAE_Jul_14_2023_14_28',
                    '/home/akinlolu/Desktop/newCode/Dirichlet-VAE-main/trained_models_MNIST/DirSparse_VAE/best_checkpoint_DirSparse_VAE_Jul_14_2023_09_00',
                    '/home/akinlolu/Desktop/newCode/Dirichlet-VAE-main/trained_models_MNIST/p_o_d_L_D_A/best_checkpoint_p_o_d_L_D_A_Jul_14_2023_12_28']


def fit_kNN_classifier(n_neighbors, features_dict):
    train_y = features_dict[xy_sets[1]].squeeze()
    n_samples = train_y.shape[0]
    train_x = features_dict[xy_sets[0]].reshape(n_samples, -1)

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(train_x, train_y)

    return classifier


def score_kNN_classifier(classifier, features_dict):
    test_y = features_dict[xy_sets[3]].squeeze()
    n_samples = test_y.shape[0]
    test_x = features_dict[xy_sets[2]].reshape(n_samples, -1)

    score = classifier.score(test_x, test_y)

    return score


def get_kNN_test_error(features_dict, n_neighbors):
    classifier = fit_kNN_classifier(n_neighbors=n_neighbors, features_dict=features_dict)
    score = score_kNN_classifier(classifier, features_dict=features_dict)
    error = 1 - score

    return error


#Load model state
def load_model(checkpoint_path):
    if 'G_D' in checkpoint_path:
        model = GD().cuda() if CUDA else GD()
    elif 'GD_Base_VAE' in checkpoint_path:
        model = GD_Base_VAE().cuda() if CUDA else GD_Base_VAE()
    elif 'DirSparse_VAE' in checkpoint_path:
        model = DirSparse_VAE().cuda() if CUDA else DirSparse_VAE()
    elif 'p_o_d_L_D_A' in checkpoint_path:
        model = podLDA().cuda() if CUDA else podLDA()

    model_state_dict = torch.load(checkpoint_path)['model_state_dict'] #model_state_dict is a field in the saved model
    model.load_state_dict(model_state_dict)

    return model


def get_models_dict(train_data, test_data):
    # create nested dict
    models_dict = dict(zip(model_names, [{} for x in model_names]))

    # get data and labels
    train_data = train_data.reshape(-1, 1, *input_shape)[:train_valid_test_splits[0]]
    test_data = test_data.reshape(-1, 1, *input_shape)
    train_labels = train_dataset.targets[:train_valid_test_splits[0]]
    test_labels = test_dataset.targets

    if CUDA:
        train_data = train_data.cuda()
        test_data = test_data.cuda()

    # get raw data features
    features_dict = dict(zip(xy_sets, [train_data.cpu(), train_labels.cpu(),
                                       test_data.cpu(), test_labels.cpu()]))
    models_dict['RawPixels'] = features_dict

    # get latent space data features
    for checkpoint_path in checkpoint_paths:
        model = load_model(checkpoint_path)
        model_name = [x for x in model_names if x in checkpoint_path][0]

        latent_train_data = model.reparameterize(*model.encode(train_data))
        latent_test_data = model.reparameterize(*model.encode(test_data))
        
        features_dict = dict(zip(xy_sets, [latent_train_data.detach().cpu().numpy(), train_labels.detach().cpu().numpy(),
                                           latent_test_data.detach().cpu().numpy(), test_labels.detach().cpu().numpy()]))
        models_dict[model_name] = features_dict

    return models_dict


def main():
    KNN_results = {'model_name': [], 'test_Error':[]}    
    models_dict = get_models_dict(train_data, test_data)
    for model in model_names:
        (KNN_results['model_name']).append(model)  
        if model in models_dict.keys():
            for n_neighbors in k:
                print(f'\nFitting and scoring {n_neighbors}-neighbor kNN trained on {model}...')
                model_error = get_kNN_test_error(models_dict[model], n_neighbors)
                (KNN_results['test_Error']).append(model_error)
                print(model_error)
    with open('knnResults/results.json', 'w', encoding='utf-8') as m:
        json.dump(KNN_results, m, ensure_ascii=False, indent=4)
    print('*' * 20)
    print(KNN_results)
if __name__ == '__main__':
    main()
