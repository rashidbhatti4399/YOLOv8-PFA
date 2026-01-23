# YOLOv8-PFA: Parallel Fusion Attention for Underwater Object Detection

This repository provides the official implementation of **YOLOv8-PFA**, a modified YOLOv8-based object detector designed for underwater object detection.  
he proposed model introduces **three targeted modifications** to improve detection robustness while maintaining computational efficiency.

This code is released to support **reproducibility**, **verification**, and **community adoption**.

---

## 📌 Method Overview

To better suit underwater degradation conditions, YOLOv8-PFA introduces three targeted modifications to the YOLOv8n baseline:

1. **Parallel Fusion Attention (PFA)**  
   The PFA module jointly enhances **channel-wise semantics** and **spatial localization** by computing channel and spatial attention **in parallel from the same input feature map**.  
   This design strengthens feature discrimination for **small, occluded, and low-contrast objects** while avoiding the information attenuation commonly observed in sequential attention mechanisms.
   - Inserted after the SPPF block  
   - Input channels: 256  
   - Channel reduction ratio: 16  

2. **Depthwise Separable Convolutions (DWConv)**  
   Depthwise separable convolutions are selectively integrated into the **backbone and neck** to reduce computational cost.  
   This significantly lowers the parameter count and GFLOPs while preserving detection accuracy.

3. **Wise-IoU v3 (WIoUv3) Loss**  
   The **WIoUv3 regression loss** is adopted during training to stabilize bounding-box optimization under **noisy and low-IoU conditions**, which are common for small and blurred underwater targets.

---
## 📊 Dataset

This implementation uses the URPC2020 public underwater object detection dataset.
- Please download the dataset from the official source and update the dataset path in: configs/urpc2020.yaml

## 🚀 Training

To train YOLOv8-PFA on URPC2020: 
python src/train.py \
  --model configs/yolov8-PFA.yaml \
  --data configs/urpc2020.yaml \
  --epochs 140 \
  --batch 32 \
  --imgsz 640 \
  --optimizer AdamW

Training settings:

- Epochs: 140
- Batch size: 32
- Optimizer: AdamW
- Initial learning rate: 4e-4
- LR scheduler: Cosine
- AMP: enabled
- Loss: WIoUv3 for bounding-box regression
- Data augmentation: RandAugment, Mosaic, HSV jitter, random scaling, translation, and erasing


## 🔁 Reproducibility

This repository provides all necessary components to reproduce the experiments reported in the paper, including:
- Complete model implementation  
- Custom modules (PFA and DWConv)  
- Training and evaluation scripts  
- Configuration files  
- Pretrained weights  
Using the provided scripts and configuration files, all results in the paper can be reproduced.

## 📚 Citation

If you use this code or model in your research, please cite our paper:
@article{your_paper_2025,
  title={YOLOv8n-PFA: A Parallel Fusion Attention Network for Enhanced Underwater Target Detection in Challenging Environments},
  author={Muhammad Rashid et al.},
  journal={Frontier in Marine Science},
  year={2026}
}



