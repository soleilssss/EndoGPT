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
import time

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

def count_parameters(model):
    """
    计算模型的参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小（假设float32，每个参数4字节）
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    print(f"模型参数量统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小: {model_size_mb:.2f} MB")
    
    return total_params, trainable_params, model_size_mb

def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    # 解析本脚本的附加参数（仅 batch_size），将其他参数交给原有解析器
    cli_args = args if args is not None else sys.argv[1:]
    bs_parser = argparse.ArgumentParser(add_help=False)
    bs_parser.add_argument('--batch_size', type=int, default=1, help='推理时的批大小')
    bs_args, remaining_cli = bs_parser.parse_known_args(cli_args)
    batch_size = max(1, bs_args.batch_size)

    opt, cmdline_args = load_opt_command(remaining_cli)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = '/data1/whl/unify/output/sun_seg_l/xdecoder_focalt_lang.yaml_conf~/run_6/00048840/default/model_state_dict.pt'


    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    
    # 计算模型参数量
    count_parameters(model)

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

    # FPS 统计变量
    total_frames = 0
    total_model_infer_seconds = 0.0
    
    for subdata in ['easy','hard']:
        # output_root = '/data1/whl/unify/evaluate/output/sun_seg_l/run_6_00048840/{}/'.format(subdata)
        # os.makedirs(output_root,exist_ok=True)
        image_pth = '/data1/whl/Datasets/xdecoder_data/SUN-SEG/test_{}'.format(subdata)
        
        with torch.no_grad():
            image_folder = os.path.join(image_pth,'images')
            files = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
            files.sort()

            # 按批次推理
            for start_idx in range(0, len(files), batch_size):
                batch_files = files[start_idx:start_idx + batch_size]

                batch_inputs = []
                image_ori_list = []
                size_list = []  # (width, height)

                for file in batch_files:
                    image_ori = Image.open(file).convert("RGB")
                    width = image_ori.size[0]
                    height = image_ori.size[1]
                    image = transform(image_ori)
                    image = np.asarray(image)
                    image_ori_np = np.asarray(image_ori)
                    images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

                    batch_inputs.append({'image': images, 'height': height, 'width': width})
                    image_ori_list.append(image_ori_np)
                    size_list.append((width, height))

                # 仅统计模型前向推理时间（同步保证计时准确）
                torch.cuda.synchronize()
                _infer_start = time.time()
                outputs = model.forward(batch_inputs)
                torch.cuda.synchronize()
                infer_elapsed = time.time() - _infer_start

                total_model_infer_seconds += infer_elapsed
                total_frames += len(batch_files)

                # 逐样本后处理（保持原有逻辑）
                # for idx_in_batch, out in enumerate(outputs):
                #     image_ori_np = image_ori_list[idx_in_batch]
                #     visual = Visualizer(image_ori_np, metadata=metadata)
                #     sem_seg = (~(out['sem_seg']>0.5)).squeeze().type(torch.uint8)
                #     sem_seg = sem_seg.cpu().numpy()

                #     # 后处理，去掉图像周围的白边
                #     border_width = 5
                #     w, h = sem_seg.shape
                #     for x in range(w):
                #         for y in range(border_width):
                #             sem_seg[x, y] = 0
                #             sem_seg[x, h - y - 1] = 0

                #     for y in range(h):
                #         for x in range(border_width):
                #             sem_seg[x, y] = 0
                #             sem_seg[w - x - 1, y] = 0

                #     pred = Image.fromarray((sem_seg)*255)
                #     # 可按需保存预测：
                #     # pred.save(os.path.join(output_root, os.path.basename(batch_files[idx_in_batch])))

    # 打印FPS统计结果
    if total_model_infer_seconds > 0 and total_frames > 0:
        fps = total_frames / total_model_infer_seconds
        print(f"推理统计: 共处理 {total_frames} 张图像, 模型前向总耗时 {total_model_infer_seconds:.3f} 秒, 平均FPS(仅模型前向) = {fps:.2f}, 批大小 = {batch_size}")
    else:
        print("未统计到有效的推理时间或帧数。")


if __name__ == "__main__":
    main()
    sys.exit(0)

# CUDA_VISIBLE_DEVICES=1 python inference/infer_sun_seg.py evaluate --conf_files configs/xdecoder/xdecoder_focall_my1_sun_seg.yaml