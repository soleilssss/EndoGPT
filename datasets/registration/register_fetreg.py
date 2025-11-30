import os
from detectron2.structures import BoxMode
import json
import cv2
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets import register_coco_instances
from utils.constants import FETREG_CLASSES

# def get_mydataset_dicts(img_dir):
#     json_file = os.path.join(img_dir, "segmentation_poly_train.json")
#     with open(json_file) as f:
#         imgs_anns = json.load(f)

#     dataset_dicts = []
#     for idx, v in enumerate(imgs_anns.values()):
#         record = {}
        
#         filename = os.path.join(img_dir, v["filename"])
#         height, width = cv2.imread(filename).shape[:2]
        
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width
      
#         annos = v["regions"]
#         objs = []
#         for _, anno in annos.items():
#             assert not anno["region_attributes"]
#             anno = anno["shape_attributes"]
#             px = anno["all_points_x"]
#             py = anno["all_points_y"]
#             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#             poly = [p for x in poly for p in x]

#             obj = {
#                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": [poly],
#                 "category_id": 0,
#             }
#             objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#     return dataset_dicts

# def register_mypolyp_semseg(root):

#     # for d in ["train", "val"]:
#     #     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     #     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
#     d = 'train'
#     DatasetCatalog.register("mypolyp_" + d, get_mydataset_dicts(root+"/" + d))
#     MetadataCatalog.get("mypolyp_" + d).set(thing_classes=["polyp"])
#     #balloon_metadata = MetadataCatalog.get("balloon_train")

# def register_mypolyp_semseg(root):

#     root = os.path.join(root, "polyseg")
#     # name = 'train'
#     # # for name, dirname in [("train", "training"), ("val", "validation")]:
#     image_dir = os.path.join(root, "train/poly_train2023")
#     gt_dir = os.path.join(root, "train/annotations")
#     json_dir = os.path.join(root, "train/segmentation_poly_train.json")
#     # name = f"mypolyp_sem_seg_{name}"

#     # DatasetCatalog.register(
#     #     name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
#     # )
#     # MetadataCatalog.get(name).set(
#     #     stuff_classes='polyp',
#     #     image_root=image_dir,
#     #     sem_seg_root=gt_dir,
#     #     evaluator_type="sem_seg",
#     #     ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
#     # )
#     register_coco_instances("mypolyp_sem_seg_train", {}, json_dir, image_dir)


def register_mypolyp_semseg(root):

    root = os.path.join(root, "FetReg")
    
    for name, dirname in [("train", "training"), ("test", "test")]:
        image_dir = os.path.join(root, name,"images")
        gt_dir = os.path.join(root, name,"annotations")
        #json_dir = os.path.join(root,name,f"segmentation_poly_{name}.json")
        name = f"fetreg_{name}"

        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=FETREG_CLASSES,
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            semseg_loader = "gray_multiclass",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )
        #register_coco_instances("mypolyp_sem_seg_train", {}, json_dir, image_dir)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_mypolyp_semseg(_root)