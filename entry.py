# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import torch
import logging
import wandb
import time
from datetime import timedelta

from utils.arguments import load_opt_command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_wandb(args, job_dir, entity='xueyanz', project='xdecoder', job_name='tmp'):
    wandb_dir = os.path.join(job_dir, 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    runid = None
    if os.path.exists(f"{wandb_dir}/runid.txt"):
        runid = open(f"{wandb_dir}/runid.txt").read()

    wandb.init(project=project,
            name=job_name,
            dir=wandb_dir,
            entity=entity,
            resume="allow",
            id=runid,
            config={"hierarchical": True},)

    open(f"{wandb_dir}/runid.txt", 'w').write(wandb.run.id)
    wandb.config.update({k: args[k] for k in args if k not in wandb.config})

def format_time(seconds):
    """
    将秒数格式化为可读的时间格式
    """
    return str(timedelta(seconds=int(seconds)))

def main(args=None):
    
    '''
    [Main function for the entry point]
    1. Set environment variables for distributed training.
    2. Load the config file and set up the trainer.
    '''

    opt, cmdline_args = load_opt_command(args)
    command = cmdline_args.command

    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir

    # update_opt(opt, command)
    world_size = 1
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

    if opt['TRAINER'] == 'xdecoder':
        from trainer import XDecoder_Trainer as Trainer
    else:
        assert False, "The trainer type: {} is not defined!".format(opt['TRAINER'])
    
    trainer = Trainer(opt)
    os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'

    if command == "train":
        if opt['rank'] == 0 and opt['WANDB']:
            wandb.login(key=os.environ['WANDB_KEY'])
            init_wandb(opt, trainer.save_folder, job_name=trainer.save_folder)
        
        # 记录训练开始时间
        train_start_time = time.time()
        logger.info(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_start_time))}")
        
        trainer.train()
        
        # 计算并输出总训练时间
        train_end_time = time.time()
        total_training_time = train_end_time - train_start_time
        
        logger.info(f"训练结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_end_time))}")
        logger.info(f"总训练时间: {format_time(total_training_time)} ({total_training_time:.2f} 秒)")
        
        # 如果使用wandb，记录训练时间
        if opt['rank'] == 0 and opt['WANDB']:
            wandb.log({"total_training_time_seconds": total_training_time})
            wandb.log({"total_training_time_formatted": format_time(total_training_time)})
        
    elif command == "evaluate":
        # Set the path of pretrained weights for evaluation
        opt['RESUME_FROM'] = 'pretrain_weights/EndoGPT_pretrained_weights.pt'
        trainer.eval()
    else:
        raise ValueError(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
    sys.exit(0)
