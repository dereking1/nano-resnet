# Plotting Functions

import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

import torch


def show_images(X, actual, title, batch_size, save=False, predicted=None):
    plt.clf()
    fig = plt.figure(f"{title}")

    # loop over the batch size
    for i in range(0, batch_size):
        # create a subplot
        ax = plt.subplot(4, 4, i+1)
        # grab the image, convert it from channels first ordering to
        # channels last ordering, and scale the raw pixel intensities
        # to the range [0, 255]
        image = X[0][i].cpu().numpy()
        image = image.transpose((1, 2, 0))
        image = (image * 255.0).astype("uint8")
        # grab the label id and get the label from the classes list

        idx = X[1][i]
        label = actual[idx]
                
        # show the image along with the label
        plt.imshow(image)
        plt.title(label)
        plt.axis("off")
        
    # show the plot
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(f'figs/{title}.png')


def error_per_epoch_all_layer(N, colors, prefix, save=False):
    plt.clf()
    dfs = [pd.read_csv(f'{prefix}{6*n +2}.csv') for n in N]
    fig = plt.figure(figsize=(40,20))
    plt.rcParams.update({'font.size': 35})

    prefix = prefix.split('/')[-1]
    title=f'All layer sizes and epochs vs. errors for {prefix} results'
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('error (%)')

    for n, c, df in zip(N, colors, dfs):
        plt.plot(df['epoch'], df['train_err']*100, f'{c}--', label=f'{prefix}-{6*n+2} train')
        plt.plot(df['epoch'], df['test_err']*100, f'{c}', label=f'{prefix}-{6*n+2} test')

    # replicate dashed lines at 0, 5, 10, and 20
    hlines = [0, 5,10,20, 30, 40]
    for h in hlines:
        plt.axhline(h, color='black', alpha=0.5, dashes=(10., 10.));        

    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper right')
    plt.tight_layout()
    if save:
        os.makedirs('figs', exist_ok=True)    
        plt.savefig(f'figs/{prefix}')


def layersize_vs_errors (layer_sizes, colors, prefix, save=False):
    plt.clf()
    dfs = [pd.read_csv(f'{prefix}{l}.csv') for l in layer_sizes]
    fig = plt.figure(figsize=(40,20))
    plt.rcParams.update({'font.size': 35})

    # NOTE: this does chooses index of the best training error and the best testing
    # error from the same epoch as the best trained error. 
    min_train_errs = [ np.argmin(df['train_err']) for df in dfs ]

    test_errs = [df['test_err'][mte] for df, mte in zip(dfs, min_train_errs)]
    train_errs = [df['train_err'][mte] for df, mte in zip(dfs, min_train_errs)]

    prefix = prefix.split('/')[-1]
    title=f'Layer sizes vs. (best training) errors for {prefix} results'
    plt.title(title)
    plt.xlabel('layer_size')
    plt.ylabel('error (%)')

    plt.plot(layer_sizes, [e*100 for e in test_errs], f'r--', label=f'test')
    plt.plot(layer_sizes, [e*100 for e in train_errs], f'g--', label=f'train')

    plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper right')
    plt.tight_layout()
    if save:
        os.makedirs('figs', exist_ok=True)    
        plt.savefig(f'figs/{prefix}_best_train_ckpt')



def plot_confusion_matrices(N, prefix, dataloader, models, classes, save=False):
    plt.clf()
    plt.rcParams.update({'font.size': 12})
    for n, model in zip(N, models):
        y_pred = []
        y_true = []

        # iterate over test dataset
        for inputs, labels in dataloader:
            output = model(inputs)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction 

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save GT
        
        # create cfsn matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                            columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)

        prefix = prefix.split('/')[-1]
        title=f'{prefix}{6*n+2}-layer Confusion Matrix for CIFAR10 Test Set'
        plt.title(title)

        if (save):
            os.makedirs('figs', exist_ok=True)    
            plt.savefig(f'figs/{prefix}{6*n+2}_CM.png')


def plot_losses(title, losses, save=False):
    plt.clf()
    plt.scatter(np.arange(len(losses)), losses)
    plt.title(title)
    plt.xlabel ( "epochs" )
    plt.ylabel ( "loss" )
    if (save):
        os.makedirs('figs', exist_ok=True)
        plt.savefig(f'figs/{title}.png')
    else:
        plt.show()

