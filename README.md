# YOLOv8-PFA: Parallel Fusion Attention for Underwater Object Detection

This repository provides the official implementation of **YOLOv8-PFA**, a modified YOLOv8-based object detector designed for underwater object detection.  
he proposed model introduces **three targeted modifications** to improve detection robustness while maintaining computational efficiency.

This code is released to support **reproducibility**, **verification**, and **community adoption**.

---

## 📌 Method Overview

To better suit underwater degradation conditions, YOLOv8-twin-OD introduces three targeted modifications to the YOLOv8n baseline:

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

## 📂 Repository Structure
