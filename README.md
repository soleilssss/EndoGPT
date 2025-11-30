## EndoGPT: A Multi-Modal Multi-Task Foundation Model for Generalist Endoscopic Analysis 

<div align="center">
  <video src="assets/UI_Function_Demo.mp4" width="100%" controls autoplay></video>
</div>

We present EndoGPT, an open-source generalist foundation model trained on 6 million endoscopic images across multiple modalities. EndoGPT employs a unified framework that converts heterogeneous annotations into a common representation, enabling classification, detection, and segmentation within a single architecture. 


## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/soleilssss/EndoGPT
cd EndoGPT
```

### 2. Create Environment
We recommend using Conda to manage the environment.

```bash
conda create -n endogpt python=3.10 -y
conda activate endogpt
```

For detailed dependency installation, please refer to [INSTALL.md](assets/readmes/INSTALL.md).

## 📂 Dataset Preparation

### Directory Structure
Please organize your datasets as follows:

```
.xdecoder_data
├── GIANA/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   └── test/
│       ├── images/
│       └── annotations/
└── SUN-SEG/
    ├── train/
        ├── images/
        ├── annotations/
    ├── test-easy/
        ├── images/
        ├── annotations/
    └── test-hard/
        ├── images/
        ├── annotations/
```

### Registering New Datasets
To add a new dataset, you need to modify the following files:
1.  **Registration**: Add registration logic in `datasets/registration/`.
2.  **Build**: Update `datasets/build.py`.
3.  **Misc Utils**: Update `modeling/utils/misc.py` and `pipeline/utils/misc.py`.
4.  **Constants**: Define class names in `utils/constants.py`.
5.  **Mapper**: Adjust the number of classes in `datasets/mappers/dataset_mapper.py` if necessary.

### Environment Variables
Before running the code, export the dataset paths:

```bash
export DETECTRON2_DATASETS=/pth/to/xdecoder_data
export DATASET=/pth/to/xdecoder_data
export DATASET2=/pth/to/xdecoder_data
# If using COCO Caption evaluation
export PATH=$PATH:/pth/to/xdecoder_data/coco_caption/jre1.8.0_321/bin
export PYTHONPATH=$PYTHONPATH:/pth/to/xdecoder_data/coco_caption
```

## 🚀 Usage

### Training
To train the model on a distributed system (e.g., 4 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 entry.py train --conf_files configs/xdecoder_focall_my1_giana.yaml
```
> **Note**: When switching between single-GPU and multi-GPU training, you may need to adjust `trainer/utils/mpi_adapter.py`.

### Inference & Visualization
To run inference and visualize results:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_giana.py evaluate --conf_files configs/xdecoder_focall_my1_giana.yaml
```
> **Note**: You may need to modify the `pretrained_pth` and `output_root` variables in `inference/infer_giana.py` before running.

### Computing Metrics
To compute evaluation metrics, use the `inference/CalculateIndicators.py` script. 

> **Note**: Make sure to set the appropriate paths for predictions and ground truth annotations within the script before running it.

## 🙏 Acknowledgement
We build our work on top of [X-Decoder](https://github.com/microsoft/X-Decoder). We thank the authors for their excellent work.
