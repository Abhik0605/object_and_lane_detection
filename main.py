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
from PIL import Image, ImageFont, ImageDraw

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

LOG = init_logger.get_logger(log_file_name_prefix='log')

def parser():
    parser = argparse.ArgumentParser(description="Lane And Object detection")
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input_vid", type=str, default=None,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--input_img", type=str, default=None,
                        help="input image path")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights_yolo", default="./darknet/weights/yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--weights_laneaf", default="./LaneAF/tusimple-weights/net_0012.pth",
                        help="laneaf weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
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


    #inference
    # darknet_image = darknet_image_queue.get()
    # prev_time = time.time()
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
    # detections_queue.put(detections)
    # fps = int(1/(time.time() - prev_time))
    # fps_queue.put(fps)

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



    if args.input_img is not None:

        img = cv2.imread(args.input_img, cv2.IMREAD_COLOR)
        
        t_start = time.time()

        img_out = laneaf(model, img)

        t_cost = time.time() - t_start
        LOG.info('Time for lane detection {:.5f}s'.format(t_cost))

        t_start = time.time()

        image_darknet = darknet_on_img(img_out)

        t_cost = time.time() - t_start
        LOG.info('Time for object detection {:.5f}s'.format(t_cost))
        while True:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow('result', image_darknet)
            if cv2.waitKey(1) & 0xFF == ord('q'):                
                break
        # combined = cv2.addWeighted(img_out, , image_darknet, 1.0, 0.0)

    # print(img_out.shape)

    # print(image_darknet.shape)

    if args.input_vid is not None:
        video_path = args.input_vid
        vid = cv2.VideoCapture(video_path)
        img_arr = []
        accum_time = 0
        curr_fps = 0
        prev_time = timer()
        while True:

            return_value, frame = vid.read()
            if not return_value:
            	break

            t_start = time.time()

            img_out = laneaf(model, frame)

            t_cost = time.time() - t_start
            LOG.info('Time for lane detection {:.5f}s'.format(t_cost))

            t_start = time.time()

            image_darknet = darknet_on_img(img_out)

            t_cost = time.time() - t_start
            LOG.info('Time for object detection {:.5f}s'.format(t_cost))

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            cv2.putText(image_darknet, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.50, color=(255, 0, 0), thickness=2)

            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow('result', image_darknet)

            img_arr.append(image_darknet)

            if cv2.waitKey(1) & 0xFF == ord('q'):                
                break

        vid.release()    
        out = cv2.VideoWriter('out/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1280, 720))
     
        for i in range(len(img_arr)):
            out.write(img_arr[i])
        out.release()