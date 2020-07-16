

import os
import random
from tqdm import tqdm
import h5py
from sklearn import metrics

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torch_models
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np

from options import Options
from models import Inception_v3
import utils


def main():
    opt = Options(isTrain=False)
    opt.parse()

    os.makedirs(opt.test['save_dir'], exist_ok=True)

    # opt.save_options()
    opt.print_options()

    # if not os.path.isfile('{:s}/test_prob_results.npy'.format(opt.test['save_dir'])):
    get_probs(opt)

    # compute accuracy
    save_dir = opt.test['save_dir']
    #print(save_dir)
    test_prob = np.load('{:s}/test_prob_results.npy'.format(save_dir), allow_pickle=True).item()

    txt_file = open('{:s}/test_results.txt'.format(save_dir), 'w')
    save_filepath = '{:s}/roc.png'.format(save_dir)
    acc, auc, auc_CIs = compute_metrics(test_prob, opt)
    #print(f'acc: {acc}, auc: {auc}, auc_CIs: {auc_CIs}')

    if opt.exp in ['fib', 'nas_lob', 'nas_balloon']:
        ave_auc = np.mean(auc)
        ave_auc_CIs = np.mean(auc_CIs, axis=0)
        #print('ave_auc_CIs', ave_auc_CIs)
        message = 'Acc: {:5.2f}\tAUC: {:5.2f} ({:5.2f}, {:5.2f})\t' \
                  'AUC0: {:5.2f} ({:5.2f}, {:5.2f})\tAUC1: {:5.2f} ({:5.2f}, {:5.2f})\tAUC2: {:5.2f} ({:5.2f}, {:5.2f})' \
            .format(acc * 100, ave_auc * 100, ave_auc_CIs[0] * 100, ave_auc_CIs[1] * 100,
                    auc[0] * 100, auc_CIs[0][0] * 100, auc_CIs[0][1] * 100,
                    auc[1] * 100, auc_CIs[1][0] * 100, auc_CIs[1][1] * 100,
                    auc[2] * 100, auc_CIs[2][0] * 100, auc_CIs[2][1] * 100)
    else:
        message = 'Acc: {:5.2f}\tAUC: {:5.2f} ({:5.2f}, {:5.2f})\n\n' \
            .format(acc * 100, auc * 100, auc_CIs[0] * 100, auc_CIs[1] * 100)

    print(message)
    txt_file.write(message)
    txt_file.close()


def get_probs(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    model_path = opt.test['model_path']
    save_dir = opt.test['save_dir']

    # create model
    model = Inception_v3(opt.model['out_c'])
    model = model.cuda()

    print("=> loading trained model")
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    # model = model.module
    print('Model obtained in epoch: {:d}'.format(best_checkpoint['epoch']))

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        }

    fold_num = opt.exp_num.split('_')[-1]
    print('Fold number: {:s}'.format(fold_num))

    #base_path = '/dresden/users/aj611/experiments/biomed/he_images/'
    if opt.exp in ['fib']:
        #base_path = '/dresden/users/aj611/experiments/biomed/he_images/'
        base_path = '/dresden/users/aj611/experiments/biomed/he_images_check/'
    else:
        #base_path = '/dresden/users/aj611/experiments/biomed/he_images_3x/'
        base_path = '/dresden/users/aj611/experiments/biomed/he_images_3x_check/'
    #data_dir = opt.train['data_dir'] + opt.exp + '/' + 'fold_{}/'.format(fold_num)
    data_dir = base_path + opt.exp + '/' + 'fold_{}/'.format(fold_num)
    print('data_dir :', data_dir)
 
    batch_size = 1
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
    #dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, drop_last=True, shuffle = True, num_workers = 8) for x in ['train', 'test']}
    dataloaders = {}
    x = 'test'
    dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size = 1, drop_last=True, shuffle = True, num_workers = 8)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

    print('data_dir :{}\n'.format(data_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------- End Data loading ---------- #


    # ----- optionally load from a checkpoint ----- #
    # if opt.train['checkpoint']:
    print("=> Test begins:")
    # switch to evaluate mode
    model.eval()
    
    prob_results = {}
    phase = 'test'
    i = 0
    for inputs, labels in dataloaders[phase]:
        slide_name = '{}'.format(i)
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            output,_ = model(inputs, is_inception=False)

        probs = nn.functional.softmax(output, dim=1).squeeze(0).cpu().numpy()
        prob_results[slide_name] = {'probs': probs, 'labels': labels.item()}
        i += 1
    #print('prob_resuls: ', prob_results)
  
    np.save('{:s}/test_prob_results.npy'.format(save_dir), prob_results)
 

def compute_metrics(slides_probs, opt):
    all_probs = []
    all_labels = []
    for slide_name, data in slides_probs.items():
        # print('{:s}\t{:.4f}\t{:.4f}\t{:.4f}\t{:d}'.format(slide_name, data['prob_nas'][0], data['prob_nas'][1],
        #                                                   data['prob_nas'][2], data['label_nas']))
        all_probs.append(data['probs'])
        all_labels.append(data['labels'])
        
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_pred = np.argmax(all_probs, axis=1).astype(np.float)

    acc = metrics.accuracy_score(all_labels, all_pred)
    if np.unique(np.array(all_labels)).size == 1:
        auc = -0.01
        auc_CIs = [-0.01, -0.01]
    else:
        if opt.exp in ['fib', 'nas_lob', 'nas_balloon']:
            auc = []
            auc_CIs = []
            for i in range(3):
                auc_i, auc_CIs_i = bootstrap_AUC_CIs(all_probs[:, i], (all_labels==i).astype(np.float))
                auc.append(auc_i)
                auc_CIs.append(auc_CIs_i)
            auc = np.array(auc)
            auc_CIs = np.array(auc_CIs)
        else:
            auc, auc_CIs = bootstrap_AUC_CIs(all_probs[:, 1], all_labels)
    return acc, auc, auc_CIs



def bootstrap_AUC_CIs(probs, labels):
    probs = np.array(probs)
    labels = np.array(labels)
    N_slide = len(probs)
    index_list = np.arange(0, N_slide)
    AUC_list = []
    i = 0
    while i < 1000:
        sampled_indices = random.choices(index_list, k=N_slide)
        sampled_probs = probs[sampled_indices]
        sampled_labels = labels[sampled_indices]

        if np.unique(sampled_labels).size == 1:
            continue

        auc_bs = metrics.roc_auc_score(sampled_labels, sampled_probs)
        AUC_list.append(auc_bs)
        i += 1

    assert len(AUC_list) == 1000
    AUC_list = np.array(AUC_list)
    auc_avg = np.mean(AUC_list)
    auc_CIs = [np.percentile(AUC_list, 2.5), np.percentile(AUC_list, 97.5)]
    return auc_avg, auc_CIs


if __name__ == '__main__':
    main()
