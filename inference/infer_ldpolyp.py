# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import yaml
import json
import argparse
import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)
sys.path.append('../')
from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from torchvision import transforms

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.visualizer import Visualizer
from utils.distributed import init_distributed

logger = logging.getLogger(__name__)

def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = '/data1/whl/unify/output/LDPolypVideo/xdecoder_focalt_lang.yaml_conf~/run_3/00041300/default/model_state_dict.pt'


    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    stuff_classes = ['polyp']
    stuff_colors = [random_color(rgb=True, maximum=255).astype(int).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)
    

    
    output_root = '/data1/whl/unify/evaluate/output/LDPolypVideo/run_3_00041300'
    os.makedirs(output_root,exist_ok=True)
    image_pth = '/data1/whl/Datasets/xdecoder_data/LDPolypVideo/test'
    
    with torch.no_grad():
        # for dataset_name in os.listdir(image_pth):
        #     dataset_folder = os.path.join(image_pth,dataset_name)
        image_folder = os.path.join(image_pth,'images')
        for file_name in os.listdir(image_folder):
            file = os.path.join(image_folder,file_name)
            image_ori = Image.open(file).convert("RGB")
            width = image_ori.size[0]
            height = image_ori.size[1]
            image = transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

            batch_inputs = [{'image': images, 'height': height, 'width': width}]
            outputs = model.forward(batch_inputs)
            visual = Visualizer(image_ori, metadata=metadata)
            #sem_seg = outputs[-1]['sem_seg'].max(0)[1]
            sem_seg = (~(outputs[-1]['sem_seg']>0.5)).squeeze().type(torch.uint8)
            
            sem_seg = sem_seg.cpu().numpy()
            # pred = Image.fromarray(sem_seg*255)
            
            
            # 后处理，去掉图像周围的白边
            border_width = 5
            width, height = sem_seg.shape
            # pixels = pred.load()
            # 上边框和下边框
            for x in range(width):
                for y in range(border_width):
                    sem_seg[x, y] = 0
                    sem_seg[x, height - y - 1] = 0

            # 左边框和右边框
            for y in range(height):
                for x in range(border_width):
                    sem_seg[x, y] = 0
                    sem_seg[width - x - 1, y] = 0
                    
            pred = Image.fromarray((sem_seg)*255)
            # import matplotlib.pyplot as plt
            # plt.imshow(sem_seg.cpu().numpy(),cmap='gray')
            # plt.show()
            # plt.savefig('/data/wkn/unify/X-Decoder-main/inference_sample.png')

            #demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image

            if not os.path.exists(output_root):
                os.makedirs(output_root)

            pred.save(os.path.join(output_root, file_name))
            
            #demo.save(os.path.join(output_root+dataset_name, file_name))


if __name__ == "__main__":
    main()
    sys.exit(0)

# CUDA_VISIBLE_DEVICES=3 python inference/infer_ldpolyp.py evaluate --conf_files configs/xdecoder/xdecoder_focall_my1_ldpolypvideo.yaml