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

from utils.arguments_my import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.visualizer import Visualizer
from utils.distributed import init_distributed
from utils.constants import M2CAISEG_CLASSES

logger = logging.getLogger(__name__)
# CHOLECSEG8K_CLASSES=["Abdominal Wall","Liver","Gastrointestinal Tract","Fat","Grasper","L-hook Electrocautery","Gallbladder"]
def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt = load_opt_command(args)
    # if cmdline_args.user_dir:
    #     absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
    #     opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = '/data1/whl/unify/output/m2caiSeg/xdecoder_focalt_lang.yaml_conf~/run_2/00001140/default/model_state_dict.pt'
    # output_root = './output/colon_lap/run_13_00151740/private/polyp/'
    output_root = '/data1/whl/unify/evaluate/output/m2caiSeg/run_2_00001140'
    os.makedirs(output_root,exist_ok=True)
    # image_pth = '/data1/whl/Datasets/endoscope-labelseg/polyp'
    image_pth = '/data1/whl/Datasets/xdecoder_data/m2caiSeg/test/images'

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    stuff_classes = M2CAISEG_CLASSES
    stuff_colors = [random_color(rgb=True, maximum=255).astype(int).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes , is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)

    with torch.no_grad():
        # for dataset_name in os.listdir(image_pth):
        #     dataset_folder = os.path.join(image_pth,dataset_name)
        image_folder =image_pth
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
            sem_seg = outputs[-1]['sem_seg'].max(0)[1]
            # sem_seg = (~(outputs[-1]['sem_seg']>0.5)).squeeze().type(torch.uint8)
            
            sem_seg = sem_seg.cpu().numpy().astype(np.uint8)
            pred = Image.fromarray(sem_seg)
            # pred = Image.fromarray((1-sem_seg)*255)

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

# CUDA_VISIBLE_DEVICES=0 python inference/infer_m2caiseg.py evaluate --conf_files /data1/whl/unify/output/m2caiSeg/xdecoder_focalt_lang.yaml_conf~/run_2/xdecoder_focall_my1_m2caiseg.yaml