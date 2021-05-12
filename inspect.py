import torch
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
from networks.poolnet import build_model, weights_init
from dataset.dataset import load_image_test
import scipy.misc as sm
import numpy as np
import torchvision.utils as vutils
import cv2

from collections import OrderedDict
import argparse
import os
import math
import time

class Inspector(object):
    def __init__(self, config):
        self.config = config
        self.build_model()
        print('Loading pre-trained model from %s...' % self.config.model_path)
        if self.config.cuda:
            self.net.load_state_dict(torch.load(self.config.model_path))
        else:
            self.net.load_state_dict(torch.load(self.config.model_path, map_location='cpu'))
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
        if self.config.cuda: self.net = self.net.cuda()
        self.net.eval()
        self.net.apply(weights_init)
        self.print_network(self.net, 'PoolNet Structure')

    def infer(self, input_path, output_path):
        input_data, _ = load_image_test(input_path)
        with torch.no_grad():
            input_data = torch.Tensor(input_data)
            input_data = torch.unsqueeze(input_data, 0)
            input_data = Variable(input_data)
            if self.config.cuda:  input_data = input_data.cuda()
        preds = self.net(input_data)
        pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
        multi_fuse = 255 * pred
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        cv2.imwrite(output_path, multi_fuse)

    def measure_fps(self, input_path):
        input_data, _ = load_image_test(input_path)
        with torch.no_grad():
            input_data = torch.Tensor(input_data)
            input_data = torch.unsqueeze(input_data, 0)
            input_data = Variable(input_data)
            if self.config.cuda:  input_data = input_data.cuda()
            preds = self.net(input_data)

        for _ in range(30):
            preds = self.net(input_data)
            pred = 255 * np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
        
        time_s = time.perf_counter()
        for _ in range(30):
            preds = self.net(input_data)
            pred = 255 * np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
        torch.cuda.synchronize()
        time_end = time.perf_counter()
        inference_time = (time_end - time_s) / 30
        print('FPS: {}'.format((1/inference_time)))
        

def main(config):
    torch.cuda.set_device(1)
    inspector = Inspector(config)
    if config.runmode == 'infer':
        inspector.infer(config.input_img_path, config.output_img_path)
    elif config.runmode == 'fps':
        inspector.measure_fps(config.input_img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runmode', type=str, choices=['infer', 'fps'], default='infer', 
    help='infer: infer from single image and write a result as a .jpg file \n fps: measure a FPS of given model')
    parser.add_argument('--arch', type=str, choices=['resnet', 'vggnet'], default='resnet', help='resnet or vgg')
    parser.add_argument('--input_img_path', metavar='DIR', required=True, help='Input image path')
    parser.add_argument('--model_path', metavar='DIR', required=True, help='.pth path to use in this demo')
    parser.add_argument('--output_img_path', metavar='DIR', required=True, 
    help='Output image path, i.e. It visualizes an inference result')
    parser.add_argument('--cpu', dest='cuda', action='store_false')
    config = parser.parse_args()
    main(config)