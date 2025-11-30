# Installation Guide

**General Environment**
* Linux System
* CUDA enabled GPU with Memory > 8GB (Evaluation)
* CUDA enabled GPU with Memory > 12GB (Training)

**Installation**

```sh
# Python Package Installation
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt

# Customer Operator [only need training deformable vision encoder]
cd modeling/vision/encoder/ops && sh make.sh && cd ../../../../

```

**Evaluation Tool**
```sh
# save coco_caption.zip to .xdecoder_data
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/coco_caption.zip
unzip coco_caption.zip
```

**Pretrained Checkpoint**

Please download the pretrained checkpoint (xdecoder_focall_bestseg.pt) from [Hugging Face](https://huggingface.co/xdecoder/X-Decoder) and place it in the `pretrain_weights/` folder.

```sh
# Example
mkdir -p pretrain_weights
# Download the checkpoint manually or via wget/git lfs
# mv /path/to/downloaded/xdecoder_focall_bestseg.pt pretrain_weights/
```