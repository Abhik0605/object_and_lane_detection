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
    parser.add_argument("--input_vid", type=str, default=None,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--input_img", type=str, default=None,
                        help="input image path")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights_yolo", default="./darknet/weights/yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--weights_laneaf", default="./LaneAF/culane-weights/net_0033.pth",
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
    parser.add_argument('--live_detection', default=False, help='get live feed from camera')

    return parser.parse_args()

#python main.py --config_file darknet\cfg\yolov4-tiny.cfg --weights_yolo darknet\weights\yolov4-tiny.weights --input_vid challenge.mp4
def laneaf(net, frame_queue, img_out_que, fps_queue, darknet_image_queue):
    net.eval()
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    video = set_saved_video(vid, f"out/video {start_time}.avi", (width, height))
            
    while vid.isOpened():
        prev_time =time.time()

        img = img_out_que.get()
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
        t_cost = time.time() - prev_time
        # fps_queue.put(t_cost)
        LOG.info('Time for lane detection {:.5f}s'.format(t_cost))
        new_tcost = fps_queue.get() + t_cost
        new_fps = int( 1/new_tcost)
        video.write(img_out)
        cv2.putText(img_out, text=f'FPS : {new_fps}', org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
           fontScale=0.50, color=(255, 0, 0), thickness=2)
        
        cv2.imshow('Inference', img_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):                
            break
        # print(frame_resized.shape)

    vid.release()
    cv2.destroyAllWindows()


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def inference(darknet_image_queue, img_out_que, detections_queue, fps_queue):
    while vid.isOpened():	
        # img_out_que.put(frame_resized)
        darknet_image = darknet_image_queue.get()

        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        # fps = int(1/(time.time() - prev_time))
        t_cost = time.time() - prev_time
        LOG.info('Time for object detection {:.5f}s'.format(t_cost))
        fps_queue.put(t_cost)
    
    vid.release()


def drawing(frame_queue, detections_queue, img_out_que,  fps_queue):	
    while vid.isOpened():
        prev_time = time.time()
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        if frame_resized is not None:
            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_out_que.put(image)

        else:
            print("Frame resized is None")
            break

    vid.release()
    # cv2.destroyAllWindows()

 
def video_capture(frame_queue, darknet_image_queue):
    while vid.isOpened():
        prev_time = time.time()
        ret, frame = vid.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(frame_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        darknet_image_queue.put(darknet_image)
    vid.release()



if __name__ == '__main__':
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    img_out_que = Queue()
    frame_queue = Queue() 
    fps_queue = Queue(maxsize=1)

    args = parser()

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

    model.cuda()

    print(model)


    if args.input_img is not None:

        img = cv2.imread(args.input_img, cv2.IMREAD_COLOR)
        
        t_start = time.time()

        img_out = Thread(target = laneaf, args = (model, img)).start()

        t_cost = time.time() - t_start
        LOG.info('Time for lane detection {:.5f}s'.format(t_cost))

        t_start = time.time()

        image_darknet = Thread(target = darknet_on_img, args = (img_out)).start()

        t_cost = time.time() - t_start
        LOG.info('Time for object detection {:.5f}s'.format(t_cost))
        while True:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow('result', img_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):                
                break
        # combined = cv2.addWeighted(img_out, , image_darknet, 1.0, 0.0)

    if args.live_detection:
        url = 'http://192.168.43.1:8080/video'
        vid = cv2.VideoCapture(url)

        Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
        Thread(target=laneaf, args=(model, frame_queue, img_out_que, fps_queue, darknet_image_queue)).start()
        Thread(target=inference, args=(darknet_image_queue, img_out_que, detections_queue, fps_queue)).start()
        Thread(target=drawing, args=(frame_queue, detections_queue, img_out_que,  fps_queue)).start()        



    if args.input_vid is not None:
        video_path = args.input_vid
        vid = cv2.VideoCapture(video_path)

        Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
        Thread(target=laneaf, args=(model, frame_queue, img_out_que, fps_queue, darknet_image_queue)).start()
        Thread(target=inference, args=(darknet_image_queue, img_out_que, detections_queue, fps_queue)).start()
        Thread(target=drawing, args=(frame_queue, detections_queue, img_out_que,  fps_queue)).start()




