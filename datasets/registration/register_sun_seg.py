import os
from detectron2.structures import BoxMode
import json
import cv2
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets import register_coco_instances


def register_mypolyp_semseg(root):

    root = os.path.join(root, "SUN-SEG")
    
    for name, dirname in [("train", "training"),('test_hard','test'),('test_easy','test'),('train_fewshot','test')]:
        image_dir = os.path.join(root, name,"images")
        gt_dir = os.path.join(root, name,"annotations")
        #json_dir = os.path.join(root,name,f"segmentation_poly_{name}.json")
        name = f"sun_seg_{name}"

        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="jpg", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=['polyp'],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            semseg_loader = "gray",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )
        #register_coco_instances("mypolyp_sem_seg_train", {}, json_dir, image_dir)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_mypolyp_semseg(_root)