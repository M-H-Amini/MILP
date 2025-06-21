import torch 
import torchvision
from torchvision.transforms import Compose, Lambda, ToTensor, Normalize
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
from torchvision.models import vgg16_bn, resnet18, vit_b_16
from torch.nn import Module, Linear, CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import mh_data as mhd 
import mh_models as mhm
import mh_optimize as mho
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import os

def experiment(ds_name, num_classes, alpha, model_names, lrs, h_init=0.01, s=1000, batch_size=1024, epochs=10, n_tries=1, device='cuda', output_name=None):
    if output_name is None:
        output_name = os.path.join('outputs', f'results_milp_{"_".join(model_names)}.csv')
    ##  Data preparation...
    t0 = time()
    ds_unlabelled, ds_fine_tuning, ds_optimization = mhd.dataSplit(ds_name, h_init=h_init, s=s)
    t1 = time()
    t_data = round(t1 - t0, 4)
    print(f'Data preparation done! It took {t_data} seconds.')
    m = len(ds_unlabelled) + len(ds_fine_tuning) + len(ds_optimization)
    ##  Fine-tuning...
    t0 = time()
    models = []
    for i, model_name in enumerate(model_names):
        print(f'Fine-tuning {model_name}...')
        model, transform, train_loss, val_loss = mhm.finetune(model_name, ds_fine_tuning, num_classes=num_classes, batch_size=batch_size, lr=lrs[i], n_epochs=epochs, device=device, n_tries=n_tries)
        models.append((model, transform))
        print(f'Training loss: {train_loss}, Validation loss: {val_loss}')
    t1 = time()
    t_fine_tuning = round(t1 - t0, 4)
    print(f'Fine-tuning done! It took {t_fine_tuning} seconds.')
    ##  Prediction on the optimization set...
    print('Predicting on the optimization set...')
    
    df_pred_optimization = predict(models, ds_optimization, batch_size=batch_size, device=device)

    # print(df_pred_optimization)
    print('Predicting on the optimization set done!')
    
    ##  MILP optimization...
    print('Optimizing...')
    t0 = time()
    effort_optimization, w_list, x_list = mho.mh_optimize(df_pred_optimization, alpha=alpha)
    t1 = time()
    t_optimization = round(t1 - t0, 4)
    print(f'Optimization done! It took {t_optimization} seconds.')
    print('Effort on the optimization set:', effort_optimization / len(df_pred_optimization))
    print('w_list:', w_list)

    ##  Prediction on the unlabelled set...
    print('Predicting on the unlabelled set...')
    t0 = time()
    df_pred_unlabelled = predict(models, ds_unlabelled, batch_size=batch_size, device=device)
    t1 = time()
    t_labelling = round(t1 - t0, 4)
    print(f'Labelling done! It took {t_labelling} seconds.')
    df_pred_unlabelled['z'] = (df_pred_unlabelled[[f'p_l_{i}' for i in range(len(models))]].nunique(axis=1) == 1).astype(int)
    df_pred_unlabelled['b'] = (df_pred_unlabelled[[f'p_l_{i}' for i in range(len(models))] + ['y']].nunique(axis=1) == 1).astype(int)
    theta_list = [df_pred_unlabelled[f'p_theta_{i}'] for i in range(len(models))]
    f = np.sum([w_list[i] * theta_list[i] for i in range(len(models))], axis=0) - 1
    df_pred_unlabelled['f'] = f
    effort_unlabelled = len(df_pred_unlabelled) - np.sum(df_pred_unlabelled['z'] * (f>0))  ##  Unnormalized effort
    accuracy_unlabelled = np.sum(df_pred_unlabelled['b'] * (f>0)) + effort_unlabelled  ##  Unnormalized accuracy
    print('Effort on the unlabelled set:', effort_unlabelled)
    print('Accuracy on the unlabelled set:', accuracy_unlabelled)
    effort_total = (len(ds_fine_tuning) + len(ds_optimization) + effort_unlabelled) / m
    accuracy_total = (len(ds_fine_tuning) + len(ds_optimization) + accuracy_unlabelled) / m 
    print('Total effort:', effort_total)
    print('Total accuracy:', accuracy_total)
    z_ratio = np.sum(df_pred_unlabelled['z']) / len(df_pred_unlabelled)
    f_ratio = np.sum(f > 0) / len(df_pred_unlabelled)

    incorrect_name = os.path.basename(output_name).replace('.csv', '')
    incorrect_folder = os.path.join('outputs', 'incorrect')
    os.makedirs(incorrect_folder, exist_ok=True)
    n_incorrect_folder = len(os.listdir(incorrect_folder))
    incorrect_folder = os.path.join(incorrect_folder, f'{incorrect_name}_{n_incorrect_folder}')
    os.makedirs(incorrect_folder, exist_ok=True)

    print('Incorrect images will be saved at:', incorrect_folder)
    ##  Pick incorrect images: f > 0 and z = 1 and b = 0...
    incorrect_indices = df_pred_unlabelled[(f > 0) & (df_pred_unlabelled['b'] == 0) & (df_pred_unlabelled['z'] == 1)].index
    ##  Save incorrect images and their predictions in a csv file...
    df_incorrect = df_pred_unlabelled.loc[incorrect_indices].reset_index(drop=True)
    df_incorrect.to_csv(os.path.join(incorrect_folder, 'predictions.csv'))
    unnormalize = Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )
    for i, idx in enumerate(tqdm(incorrect_indices)):
        x = unnormalize(ds_unlabelled[idx][0]).numpy()
        x = np.transpose(x, (1, 2, 0))
        x = (x * 255).astype(np.uint8)
        plt.imsave(os.path.join(incorrect_folder, f'{i}.png'), x)
    print('Incorrect images saved at:', incorrect_folder)
    ##  Output...
    columns = ['dataset', 'size', 'alpha', 'models', 'h_init', 's', 'batch_size', 'epochs', 'w', 't_data', 't_fine_tuning', 't_optimization', 't_labelling', 'condition1_ratio', 'condition2_ratio', 'effort_optimization', 'effort_unlabelled', 'accuracy_unlabelled', 'effort_total', 'accuracy_total']
    values = [ds_name, m, alpha, model_names, h_init, s, batch_size, epochs, w_list, t_data, t_fine_tuning, t_optimization, t_labelling, z_ratio, f_ratio, effort_optimization/len(ds_optimization), effort_unlabelled/len(ds_unlabelled), accuracy_unlabelled/len(ds_unlabelled), effort_total, accuracy_total]
    if os.path.exists(output_name):
        df_output = pd.read_csv(output_name)
        df_output.loc[len(df_output)] = values
    else:
        os.makedirs('outputs', exist_ok=True)
        df_output = pd.DataFrame(columns=columns)
        df_output.loc[0] = values
    df_output.to_csv(output_name, index=False)
    print('Output saved at:', output_name)

        
def predict(models, ds, batch_size=512, device='cuda'):
    pred_cols = [f'p_l_{i}' for i in range(len(models))] + [f'p_theta_{i}' for i in range(len(models))] + ['y']
    df_pred = pd.DataFrame(columns=pred_cols)
    for i, (model, transform) in enumerate(models):
        model.eval()
        ds.transform = transform
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        pred_l = []
        pred_theta = []
        y_list = []

        for x, y in tqdm(dl):
            x = x.to(device)
            y_list.extend(y.cpu().numpy().tolist())
            with torch.no_grad():
                y_hat = model(x)
                l = torch.argmax(y_hat, dim=1).cpu().numpy().tolist()
                theta = torch.nn.functional.softmax(y_hat, dim=1).max(dim=1).values.cpu().numpy().tolist()
                pred_l.extend(l)
                pred_theta.extend(theta)
        
        df_pred[f'p_l_{i}'] = pred_l
        df_pred[f'p_theta_{i}'] = pred_theta
        df_pred[f'y'] = y_list
    
    return df_pred


if __name__ == '__main__':
    datasets = ['svhn', 'mnist', 'fashionmnist', 'cifar10'][::-1]
    # datasets = ['svhn', 'mnist', 'fashionmnist', 'cifar10']
    # datasets = ['mnist', 'fashionmnist']
    # datasets = ['cifar10', 'svhn']
    # datasets = ['fashionmnist', 'cifar10', 'svhn', 'mnist']
    # datasets = ['svhn', 'mnist'][::-1]
    # datasets = ['svhn']
    # datasets = ['cifar10', 'fashionmnist']
    # datasets = ['hu_cifar10', 'hu_imagenet']
    n_classes = 10
    h_values = [0.01, 0.05, 0.15, 0.25, 0.35, 0.45]
    # h_values = [0.15, 0.25, 0.35]
    alpha = 0.9
    repeats = 5
    epochs = 50
    batch_size = 256
    n_tries = 3  ##  Number of tries for fine-tuning
    model_names = ['resnet', 'vgg', 'vit']
    lrs = [3e-3, 1e-3, 3e-4]
    # model_names = ['resnet', 'vgg']
    # lrs = [3e-3, 1e-3]
    
    # model_names = ['resnet']
    # lrs = [3e-3]
    output_name = os.path.join('outputs', f'results_milp_{alpha}_{"_".join(model_names)}_{"_".join(datasets)}.csv')
    for ds_name in datasets:
        for h_init in h_values:
            for i in range(repeats):
                print(f'Experiment on {ds_name} with alpha={alpha} and h_init={h_init} and repeat={i+1}...')
                experiment(ds_name, n_classes, alpha, model_names, lrs, h_init=h_init, s=500, epochs=epochs, batch_size=batch_size, n_tries=n_tries, output_name=output_name)
    