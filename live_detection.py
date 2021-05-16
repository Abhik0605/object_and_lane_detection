import os
import json
from datetime import datetime
from statistics import mean
import argparse

import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from LaneAF.models.dla.pose_dla_dcn import get_pose_net
from LaneAF.utils.affinity_fields import decodeAFs
from LaneAF.utils.metrics import match_multi_class, LaneEval
from LaneAF.utils.visualize import tensor2image, create_viz

import matplotlib.pyplot as plt

import torch

import LaneAF.datasets.transforms as trnsf 

from timeit import default_timer as timer

from ctypes import *
import random

import time

import darknet.darknet as darknet

import argparse

from threading import Thread, enumerate

from queue import Queue

from log_utils import init_logger

import requests

from multiprocessing import  Process

LOG = init_logger.get_logger(log_file_name_prefix='log')

def parser():
    parser = argparse.ArgumentParser(description="Lane And Object detection")
    parser.add_argument("--weights_yolo", default="./darknet/weights/yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--weights_laneaf", default="./LaneAF/culane-weights/net_0033.pth",
                        help="laneaf weights path")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./darknet/cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./darknet/cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='do not use cuda for training')

    return parser.parse_args()


def laneaf(net, img):
    net.eval()
    img = img.astype(np.float32)/255
    img=cv2.resize(img[14:,:,:], (1664, 576), interpolation=cv2.INTER_LINEAR)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_transforms = transforms.Compose([
            trnsf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            trnsf.GroupNormalize(mean=([0.485, 0.456, 0.406], (0, )), std=([0.229, 0.224, 0.225], (1, ))),
        ])
    input_img,_ = img_transforms((img,img))
    input_img = torch.from_numpy(input_img).permute(2,0,1).contiguous().float()
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
    # input_img = tf.Variable(torch.tensor(input_img))
    input_img = torch.tensor(input_img).cuda()
    outputs = net(input_img)[-1]

    img = tensor2image(input_img.detach(), np.array([0.485, 0.456,0.406]), np.array([0.229 ,0.224, 0.225]))       
    mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(), 
        np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
    vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))       
    # decode AFs to get lane instances
    seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=5)

    img_out = create_viz(img, seg_out.astype(np.uint8), mask_out, vaf_out, haf_out)
    img_out = cv2.resize(img_out, (1280, 720), interpolation=cv2.INTER_LINEAR)

    return img_out




def darknet_on_img(img):
    # get image
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frame_resized = cv2.resize(frame_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    # frame_queue.put(frame_resized)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    # darknet_image_queue.put(darknet_image)
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)

    #draw boxes

    darknet.print_detections(detections, args.ext_output)

    image = darknet.draw_boxes(detections, frame_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


if __name__ == '__main__':
    args = parser()

    prev_timer = timer()
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights_yolo,
        batch_size=1
    )

    width = 1280
    height = 720
    darknet_image = darknet.make_image(width, height, 3)



    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model.load_state_dict(torch.load(args.weights_laneaf), strict=True)

    if args.cuda:
        model.cuda()


    url = 'http://192.168.43.1:8080/shot.jpg'
    save_vid = []
    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    while True:
        t_start = time.time()

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
        img = cv2.imdecode(img_arr, -1)

        t_cost_1 = time.time() - t_start
        LOG.info('Time to get live image {:.5f}s'.format(t_cost_1))


        t_start = time.time()

        img_out = laneaf(model, img)

        t_cost_2 = time.time() - t_start
        LOG.info('Time for lane detection {:.5f}s'.format(t_cost_2))

        t_start = time.time()

        image_darknet = darknet_on_img(img_out)

        t_cost_3 = time.time() - t_start
        LOG.info('Time for object detection {:.5f}s'.format(t_cost_3))

        fps = str(int(1/(t_cost_1 + t_cost_2 + t_cost_3)))

        cv2.putText(image_darknet, text=f'FPS : {fps}', org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow('result', image_darknet)
        save_vid.append(image_darknet)

        if cv2.waitKey(1) & 0xFF == ord('q'):                
            break
            vid.release()    

    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    out = cv2.VideoWriter('out/project {:s}.avi'.format(start_time),cv2.VideoWriter_fourcc(*'DIVX'), 15, (1280, 720))
 
    for i in range(len(save_vid)):
        out.write(save_vid[i])
    out.release()

