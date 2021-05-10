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

from datasets.tusimple import TuSimple, get_lanes_tusimple
from models.dla.pose_dla_dcn import get_pose_net
from utils.affinity_fields import decodeAFs
from utils.metrics import match_multi_class, LaneEval
from utils.visualize import tensor2image, create_viz
from PIL import Image, ImageFont, ImageDraw

import matplotlib.pyplot as plt

import torch

import datasets.transforms as trnsf 

from timeit import default_timer as timer

parser = argparse.ArgumentParser('Options for inference with LaneAF models in PyTorch...')
parser.add_argument('--dataset_dir', type=str, default=None, help='path to dataset')
parser.add_argument('--output_dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--split', type=str, default='test', help='dataset split to evaluate on (train/val/test)')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no_cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--save_viz', action='store_true', default=False, help='save visualization depicting intermediate and final results')
parser.add_argument('--input_img', type=str, default=None, help='path to image')
parser.add_argument('--input_vid', type=str, default=None, help='path to video')

args = parser.parse_args()
# check args
# if args.dataset_dir is None:
#     assert False, 'Path to dataset not provided!'
# if args.snapshot is None:
#     assert False, 'Model snapshot not provided!'
# if args.split is ['train', 'val', 'test']:
#     assert False, 'Incorrect dataset split provided!'

# set batch size to 1 for visualization purposes
args.batch_size = 1

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
# print(args.cuda)
# # if args.output_dir is None:
# #     temp = datetime.now().strftime("%Y-%m-%d-%H:%M-infer")
# #     args.output_dir = os.mkdir(f'out{str(temp)}')


# # if not os.path.exists(args.output_dir):
# #     os.makedirs(args.output_dir)
# # else:
# #     assert False, 'Output directory already exists!'

# # store config in output directory
# # with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
# #     json.dump(vars(args), f)

# # set random seed
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

# # kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 1}
# # test_loader = DataLoader(TuSimple(args.dataset_dir, 'train', False), **kwargs)
# # print(enumerate(test_loader))

# # test function
# def test(net):
#     net.eval()
#     out_vid = None
#     json_pred = [json.loads(line) for line in open(os.path.join(args.dataset_dir, 'seg_label', args.split+'.json')).readlines()]

#     for b_idx, sample in enumerate(test_loader):
#         input_img, input_seg, input_mask, input_af = sample
#         # if args.cuda:
#         #     input_img = input_img.cuda()
#         #     input_seg = input_seg.cuda()
#         #     input_mask = input_mask.cuda()
#         #     input_af = input_af.cuda()

#         st_time = datetime.now()
#         # do the forward pass
#         outputs = net(input_img)[-1]

#         # convert to arrays
#         img = tensor2image(input_img.detach(), np.array(test_loader.dataset.mean), 
#             np.array(test_loader.dataset.std))
#         mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(), 
#             np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
#         vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
#         haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

#         # decode AFs to get lane instances
#         seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=5)
#         ed_time = datetime.now()

#         # re-assign lane IDs to match with ground truth
#         seg_out = match_multi_class(seg_out.astype(np.int64), input_seg[0, 0, :, :].detach().cpu().numpy().astype(np.int64))

#         # fill results in output structure
#         json_pred[b_idx]['run_time'] = (ed_time - st_time).total_seconds()*1000.
#         if json_pred[b_idx]['run_time'] > 200:
#             json_pred[b_idx]['run_time'] = 200
#         json_pred[b_idx]['lanes'] = get_lanes_tusimple(seg_out, json_pred[b_idx]['h_samples'], test_loader.dataset.samp_factor)

#         # write results to file
#         with open(os.path.join(args.output_dir, 'outputs.json'), 'a') as f:
#             json.dump(json_pred[b_idx], f)
#             f.write('\n')

#         # create video visualization

#         if args.save_viz:
#             img_out = create_viz(img, seg_out.astype(np.uint8), mask_out, vaf_out, haf_out)
#             img_out = cv2.resize(img_out, (1280, 720), interpolation=cv2.INTER_LINEAR)
#             print(img_out.shape)
#             cv2.namedWindow("result", cv2.WINDOW_NORMAL)

#             cv2.imshow("result", img_out.astype(np.uint8))
#             if out_vid is None:
#                 out_vid = cv2.VideoWriter(os.path.join(args.output_dir, 'out.mkv'), 
#                     cv2.VideoWriter_fourcc(*'H264'), 5, (img_out.shape[1], img_out.shape[0]))
#             out_vid.write(img_out)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         print('Done with image {} out of {}...'.format(min(args.batch_size*(b_idx+1), len(test_loader.dataset)), len(test_loader.dataset)))

#     # benchmark on TuSimple
#     results = LaneEval.bench_one_submit(os.path.join(args.output_dir, 'outputs.json'), os.path.join(args.dataset_dir, 'seg_label', args.split+'.json'))
#     with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
#         json.dump(results, f)

#     if args.save_viz:
#         out_vid.release()

#     return

def test_vid(net):
    net.eval()
    video_path = args.input_vid
    vid = cv2.VideoCapture(video_path)
    img_arr = []
    prev_time = timer()
    accum_time = 0
    curr_fps = 0
    while True:
        return_value, frame = vid.read()
        if not return_value:
        	break
        # img=cv2.imread(args.input_img).astype(np.float32)/255.
        # img = Image.fromarray(frame)
        try:
            frame = frame.astype(np.float32)/255
        except:
            continue

        img=cv2.resize(frame[14:,:,:], (1664, 576), interpolation=cv2.INTER_LINEAR)
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
        # print(np.array(test_loader.dataset.std))

        

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        cv2.putText(img_out, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.50, color=(255, 0, 0), thickness=2)

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)      

        cv2.imshow("result", img_out.astype(np.uint8))  
        img_arr.append(img_out.astype(np.uint8))
        print(return_value)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):                
            break

    vid.release()    
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1280, 720))
 
    for i in range(len(img_arr)):
        out.write(img_arr[i])
    out.release()

    return


if __name__ == "__main__":
    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)

    model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    test_vid(model)

