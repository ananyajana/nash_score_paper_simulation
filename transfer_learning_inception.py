from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from random import randint
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#plt.ion()
import time
from datetime import datetime

from options import Options
import utils

# using the options from code_live_miccai
opt = Options(isTrain=True)
opt.parse()
opt.save_options()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

# set up logger
logger, logger_results = utils.setup_logger(opt)
opt.print_options(logger)


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
 
base_path = '/dresden/users/aj611/experiments/biomed/he_images/'
data_dir = base_path + opt.exp_name + '/' + 'fold_{}/'.format(fold_num)
batch_size = 8
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, drop_last=True, shuffle = True, num_workers = 8) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

logger.info('data_dir :{}\n'.format(data_dir))
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


plot_train_losses = []
plot_test_losses = []
def train_model(model, criterion, optimizer, scheduler, num_epochs=1, is_inception=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logger.info('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        logger.info('\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs, is_inception)
                        m = torch.nn.Softmax(dim = 1)
                        outputs = m(outputs)
                        aux_outputs = m(aux_outputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                        
                    else:
                        outputs, _ = model(inputs, is_inception = False)
                        #print('outputs before softmax:', outputs)
                        m = torch.nn.Softmax(dim = 1)
                        outputs = m(outputs)
                        #print('outputs after softmax:', outputs)
                        loss = criterion(outputs, labels)

                    np_outputs = outputs.cpu().detach().numpy()
                    np_outputs = np_outputs[:, 1]

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'test':
                scheduler.step(loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            running_acc_thresh = {}
            if phase == 'train':
                plot_train_losses.append(epoch_loss)
            else:
                plot_test_losses.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            logger.info('\n')

            # deep copy the model
            #if phase == 'test' and epoch_acc > best_acc:
            if phase == 'test' and test_auc > best_score:
                best_score = max(test_auc, best_score)
                best_model_wts = copy.deepcopy(model.state_dict())

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
                logger_results.info('{:<6d}| {:<12.4f}{:<12.4f}||  {:<12.4f}{:<12.4f}{:<12.4f}'
                                    .format(epoch, train_loss, train_acc,
                                            test_loss, test_acc, test_auc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('\n')
    logger.info('Best val Acc: {:4f}'.format(best_acc))
    logger.info('\n')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()

inputs, classes = next(iter(dataloaders['train']))
print(inputs.size())
print(classes)


def save_checkpoint(state, is_best, save_dir, cp_flag, epoch):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if is_best:
        torch.save(state, '{:s}/checkpoint_best.pth.tar'.format(save_dir))
    # filename = '{:s}/checkpoint.pth.tar'.format(save_dir)
    # torch.save(state, filename)
    # if cp_flag:
    #     shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(save_dir, epoch))
    # if is_best:
    #     shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(save_dir))


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


class Inception_v3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.incnet = models.inception_v3(pretrained=True)
        self.fc = nn.Linear(2048, 5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

        for param in self.incnet.parameters():
            param.requires_grad = True

    def forward(self, x, is_inception = False):
        x = self.incnet.Conv2d_1a_3x3(x)
        x = self.incnet.Conv2d_2a_3x3(x)
        x = self.incnet.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        
        x = self.incnet.Conv2d_3b_1x1(x)
        x = self.incnet.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)

        x = self.incnet.Mixed_5b(x)
        x = self.incnet.Mixed_5c(x)
        x = self.incnet.Mixed_5d(x)

        x = self.incnet.Mixed_6a(x)
        x = self.incnet.Mixed_6b(x)
        x = self.incnet.Mixed_6c(x)
        x = self.incnet.Mixed_6d(x)
        x = self.incnet.Mixed_6e(x)
        
        if is_inception:
            aux = self.incnet.AuxLogits(x)
        else:
            aux = None

        x = self.incnet.Mixed_7a(x)
        x = self.incnet.Mixed_7b(x)
        x = self.incnet.Mixed_7c(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x, aux

model_ft = Inception_v3(len(class_names))

logger.info('all layers trainable')

for param in model_ft.parameters():
    param.requires_grad = True


model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# params are taken from the nas scoring paper
lr = 0.00005
logger.info('lr :{}\n'.format(lr))
logger.info('batch size:{}\n'.format(batch_size))
optimizer_ft = optim.SGD(model_ft.parameters(), lr = lr, momentum = 0.9)

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.2, patience=2, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=1e-7, eps=1e-08)

num_epochs = 2
logger.info('Number of epochs :{}\n'.format(num_epochs))
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs = num_epochs)
plt.plot(plot_train_losses)
plt.savefig('train_losses.png')
plt.plot(plot_test_losses)
plt.savefig('test_losses.png')
