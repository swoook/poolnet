import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
from networks.poolnet import build_model, weights_init
from dataset.dataset import load_image_test
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import cv2
import math
import time


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [15,]
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()
        # self.net.train()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
        if self.config.load == '':
            self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))
        else:
            self.net.load_state_dict(torch.load(self.config.load))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)
        self.print_network(self.net, 'PoolNet Structure')


    def set_parameter_requires_grad(self, is_train):
        for param in self.net.parameters():
            param.requires_grad = is_train
        return 0

    def infer(self, input_path, output_path):
        input_data, _ = load_image_test(input_path)
        preds = self.net(input_data)
        pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
        multi_fuse = 255 * pred
        cv2.imwrite(output_path, multi_fuse)

    def test(self):
        self.net.eval()
        self.set_parameter_requires_grad(is_train=False)
        num_data = 0
        mae = 0.0
        for i, data_batch in enumerate(tqdm(self.test_loader)):
            images, name= data_batch['image'], data_batch['name'][0]
            label = data_batch['label']

            if (images.size(2) != label.size(2)) or (images.size(3) != label.size(3)):
                # print('IMAGE ERROR, PASSING```')
                continue
            num_data += 1
            label = np.squeeze(label.cpu().data.numpy())
            with torch.no_grad():
                images = Variable(images)
                if self.config.cuda:
                    images = images.cuda()
                preds = self.net(images)
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                mae += mean_absolute_error(label, pred)

                #multi_fuse = 255 * pred
                #cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)
        mae /= num_data
        return mae

    def get_mae(self, pred, label):
        pred = torch.sigmoid(pred).cpu().data.numpy()
        label = label.cpu().data.numpy()
        mae = np.average(np.abs(pred - label))
        return mae

    # training phase
    def train(self):
        self.net.train()
        self.set_parameter_requires_grad(is_train=True)
        eval_res_path = '{}/models/MAE.txt'.format(self.config.save_folder)
        eval_res_dir = os.path.dirname(eval_res_path)
        if not os.path.exists(eval_res_dir): os.makedirs(eval_res_dir)
        eval_res_file = open(eval_res_path, 'w')

        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        for epoch in range(self.config.epoch):
            r_sal_loss= 0
            train_mae = 0.0
            self.net.zero_grad()
            num_train_data = 0
            for i, data_batch in enumerate(tqdm(self.train_loader)):
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    # print('IMAGE ERROR, PASSING```')
                    continue
                num_train_data += sal_image.size(0)
                sal_image, sal_label= Variable(sal_image), Variable(sal_label)
                if self.config.cuda:
                    # cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                sal_pred = self.net(sal_image)
                train_mae += self.get_mae(sal_pred, sal_label) * sal_image.size(0)
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data

                sal_loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                # if i % (self.show_every // self.config.batch_size) == 0:
                #     if i == 0:
                #         x_showEvery = 1
                #     print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                #         epoch, self.config.epoch, i, iter_num, r_sal_loss/x_showEvery))
                #     print('Learning rate: ' + str(self.lr))
                #     r_sal_loss= 0

            train_mae /= num_train_data
            test_mae = self.test()
            print('epoch: [%2d/%2d]' % (
                epoch, self.config.epoch))
            print('Learning rate: ' + str(self.lr))
            print('{:>16} {:>16} {:>16}'.format('', 'DUTS-TR', 'DUTS-TE'))
            print('{:>16} {:>16.3f} {:>16.3f}'.format('MAE', train_mae, test_mae))
            print('\n')
            eval_res_file.write('{} {}\n'.format(train_mae, test_mae))
            self.net.train()
            self.set_parameter_requires_grad(is_train=True)

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)

def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

