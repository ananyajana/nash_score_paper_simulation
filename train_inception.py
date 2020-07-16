import shutil
import time
import os
import math
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from models import Inception_v3
import numpy as np
from sklearn import metrics

from options import Options
import utils
from random import randint
import os
import copy
import time


def main():
    global best_score, logger, logger_results, slide_weights
    print('inside train_inception.py ')
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # set up logger
    logger, logger_results = utils.setup_logger(opt)
    opt.print_options(logger)

    # ---------- Create model ---------- #
    model = Inception_v3(opt.model['out_c'])
    model = model.cuda()

    # logger.info(model)
    # ---------- End create model ---------- #

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    # params are taken from the nas scoring paper
    lr = 0.00005
    logger.info('lr :{}\n'.format(lr))
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=1e-7, eps=1e-08)


    # ---------- Data loading ---------- #
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(randint(1, 45)),
            transforms.RandomAffine(degrees=0,translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        }

    fold_num = opt.exp_num.split('_')[-1]
    logger.info('Fold number: {:s}'.format(fold_num))

    if opt.exp in ['fib']:
        #base_path = '/dresden/users/aj611/experiments/biomed/he_images/'
        base_path = '/dresden/users/aj611/experiments/biomed/he_images_check/'
    else:
        #base_path = '/dresden/users/aj611/experiments/biomed/he_images_3x/'
        base_path = '/dresden/users/aj611/experiments/biomed/he_images_3x_check/'
    #data_dir = opt.train['data_dir'] + opt.exp + '/' + 'fold_{}/'.format(fold_num)
    data_dir = base_path + opt.exp + '/' + 'fold_{}/'.format(fold_num)
    print('data_dir :', data_dir)
    #data_dir = '/dresden/users/aj611/experiments/biomed/he_images_bkp/fibrosis/'


    batch_size = 8
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    #dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, drop_last=True, shuffle = True, num_workers = 8) for x in ['train', 'test']}
    dataloaders = {}
    x = 'train'
    dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, drop_last=True, shuffle = True, num_workers = 8)
    x = 'test'
    dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size = 1, drop_last=True, shuffle = True, num_workers = 8)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    logger.info('data_dir :{}\n'.format(data_dir))
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------- End Data loading ---------- #


    # ----- optionally load from a checkpoint ----- #
    # if opt.train['checkpoint']:
    #     model_state_dict, optimizer_state_dict = load_checkpoint(opt.train['checkpoint'])
    #     model.load_state_dict(model_state_dict)
    #     optimizer.load_state_dict(optimizer_state_dict)
    # ----- End checkpoint loading ----- #

    # ----- Start training ---- #
    best_score = 0
    for epoch in range(opt.train['epochs']):
        # train and validate for one epoch
        test_auc = train_model(model, opt, dataset_sizes, dataloaders, image_datasets, criterion, optimizer, exp_lr_scheduler, epoch)
        print('epoch {}'.format(epoch))
        # remember best accuracy and save checkpoint
        is_best = test_auc > best_score
        best_score = max(test_auc, best_score)
        cp_flag = False
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, opt.train['save_dir'], cp_flag, epoch+1)

        # save training results
        #logger_results.info('{:<6d}| {:<12.4f}{:<12.4f}||  {:<12.4f}{:<12.4f}{:<12.4f}'
        #                    .format(epoch, train_loss, train_acc,
        #                            test_loss, test_acc, test_auc))

    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()

    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train_model(model, opt, dataset_sizes, dataloaders, image_datasets, criterion, optimizer, scheduler, num_epochs=1, is_inception=True):
    batch_time = utils.AverageMeter()
    data_time =utils.AverageMeter()
    losses = utils.AverageMeter()
    acc = utils.AverageMeter
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    auc = 0.0

    N = 0

    print('Epoch {}'.format(num_epochs))
    print('-' * 10)
    logger.info('Epoch {}\n'.format(num_epochs))
    logger.info('-' * 10)
    logger.info('\n')

    # Each epoch has a training and validation phase
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        if phase == 'train':
            N = 8 # batch size
        else:
            N = 1
            slide_probs_all = []
            slide_targets_all = []

        end = time.time()
        running_corrects = 0

        # Iterate over data.
        i = 0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                if is_inception and phase == 'train':
                    outputs, aux_outputs = model(inputs, is_inception)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4*loss2
                    
                else:
                    outputs, _ = model(inputs, is_inception = False)
                    loss = criterion(outputs, labels)

                # measure accuracy
                probs = nn.functional.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).cpu()
                accuracy = (pred == labels.cpu()).sum().numpy()
                #accuracy = torch.from_numpy(accuracy)
                #accuracy = accuracy.cuda()
                #print('loss: ', loss)
                #print('type(loss): ', type(loss))
                #print('accucary: ', accuracy)
                #print('type(accuracy) :', type(accuracy))
                #print('loss.item() : ', loss.item())
                #print('accuracy.item() : ', accuracy.item())

                if phase == 'test':
                    slide_probs_all.append(probs.detach().cpu())
                    slide_targets_all.append(labels.detach().cpu())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                loss /= N
                accu = float(accuracy)/float(N)

                #acc.update(accuracy, N)
                losses.update(loss.item(), N)

                del outputs
                if phase == 'test':
                    del probs

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_corrects += torch.sum(pred == labels.cpu())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if phase == 'train':
                if i % opt.train['log_interval'] == 0:
                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                                'Batch Time: {batch_time.avg:.3f}\t'
                                'Loss: {loss.avg:.3f}\t'
                                'Acc: {acc:.4f}'
                                .format(num_epochs, i, dataset_sizes[phase], batch_time=batch_time,
                                        loss=losses, acc=accu))

            else:
                if i % opt.train['log_interval'] == 0:
                    logger.info('Test: [{0}][{1}/{2}]\t'
                                'Time: {batch_time.avg:.3f}\t'
                                'Loss: {loss.avg:.3f}'
                                .format(num_epochs, i, dataset_sizes[phase], batch_time=batch_time, loss=losses))
            i += 1
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        if phase == 'train':
            logger.info('=> Train Avg: Loss: {loss.avg:.3f}\t\tAcc: {acc:.4f}'
                        .format(loss=losses, acc=epoch_acc))
        if phase == 'test':
            scheduler.step(loss)
            slide_probs_all = torch.cat(slide_probs_all, dim=0).numpy()
            slide_targets_all = torch.cat(slide_targets_all, dim=0).numpy()

            pred = np.argmax(slide_probs_all, axis=1)
            acc = metrics.accuracy_score(slide_targets_all, pred)

            if opt.exp in ['fib', 'nas_lob', 'nas_balloon']:
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(3):
                    fpr[i], tpr[i], _ = metrics.roc_curve(slide_targets_all == i, slide_probs_all[:, i])
                    auc_i = metrics.auc(fpr[i], tpr[i])
                    roc_auc[i] = 0 if math.isnan(auc_i) else auc_i
                auc = np.mean(np.array(list(roc_auc.values())))

                logger.info('Test Avg: {}\tLoss: {:.3f}\tAcc: {:.4f}\tAUC: {:.4f}\n'
                            'AUC0: {:.4f}\tAUC1: {:.4f}\tAUC2: {:.4f}\n'
                            .format(num_epochs, losses.avg, epoch_acc, auc, roc_auc[0], roc_auc[1], roc_auc[2]))
            else:
                tp = np.sum((pred == 1) * (slide_targets_all == 1))
                tn = np.sum((pred == 0) * (slide_targets_all == 0))
                fp = np.sum((pred == 1) * (slide_targets_all == 0))
                fn = np.sum((pred == 0) * (slide_targets_all == 1))
                acc = metrics.accuracy_score(slide_targets_all, pred)
                auc = metrics.roc_auc_score(slide_targets_all, slide_probs_all[:, 1])

                logger.info('Test Avg: {}\tLoss: {:.3f}\tAcc: {:.4f}\tAUC: {:.4f}\n'
                            'TP: {:d}\tTN: {:d}\tFP: {:d}\tFN: {:d}\n'
                            .format(num_epochs, losses.avg, epoch_acc, auc, tp, tn, fp, fn))

    return auc

def save_checkpoint(state, is_best, save_dir, cp_flag, epoch):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if is_best:
        torch.save(state, '{:s}/checkpoint_best.pth.tar'.format(save_dir))


def load_checkpoint(checkpoint_path):
    model_state_dict = None
    optimizer_state_dict = None
    if os.path.isfile(checkpoint_path):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = checkpoint['state_dict']
        optimizer_state_dict = checkpoint['optimizer']
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(checkpoint_path))
    return model_state_dict, optimizer_state_dict


if __name__ == '__main__':
    main()
