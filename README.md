# SmartMeter-Digit-OCR: Robust End-to-End Meter Reading System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)
![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.4%2B-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **"From Pixels to Digits: Bridging the Gap Between Complex Industrial Environments and Reliable Data Governance."**

---

# 1. Introduction

SmartMeter-Digit-OCR is an end-to-end intelligent meter reading system designed to address the challenge of **high-precision automatic digit extraction from smart meters under complex industrial and natural environments**.

Conventional meter recognition approaches often suffer from performance degradation caused by illumination variations, perspective distortions, reflective surfaces, background interference, and image noise. These challenges significantly limit the reliability of automated meter reading systems in real-world deployment scenarios.

To overcome these limitations, this project proposes and implements a **Cascaded Dual-Stage Visual Architecture**, establishing a complete end-to-end inference pipeline from raw image acquisition to structured numerical output.

Beyond model development, this project emphasizes **engineering-oriented deployment and MLOps principles**, including modular architecture design, lightweight inference, model decoupling, and industrial edge deployment, enabling reliable integration into practical data acquisition and governance workflows.

---

# 2. Key Features

## 🔹 Coarse-to-Fine Cascaded Detection Framework

A YOLO-based object detection module is introduced as the first-stage perception component to accurately locate and crop meter regions of interest (ROI) from complex backgrounds.

This coarse-to-fine strategy significantly improves downstream recognition robustness by reducing background interference and enhancing digit-level feature extraction.

---

## 🔹 Sequence-Based Feature Decoding

The system integrates a CRNN-based PaddleOCR recognition architecture with spatial-frequency feature modeling.

By leveraging sequential visual representation learning and CTC-based decoding, the model achieves accurate recognition of irregular meter digit sequences under challenging imaging conditions.

---

## 🔹 Engineering-Oriented Lightweight Deployment

Following modern MLOps principles, this project separates source code from large-scale model weights.

The repository provides:

- Standardized dependency management through `requirements.txt`
- Modular inference pipeline design
- Reproducible deployment scripts
- Lightweight model loading mechanism

This design substantially reduces deployment complexity and improves reproducibility for industrial applications.

---

## 🔹 Industrial-Grade ONNX Deployment

The recognition module has been successfully exported into a framework-independent ONNX static graph.

The deployment version supports dynamic tensor width adaptation:

```text
[-1, 3, 48, -1]
```

On a real-world blind evaluation dataset containing **5,000 complex industrial images**, the deployed model achieves:

```text
Recognition Accuracy: 99.64%
```

demonstrating strong robustness and practical deployment capability for edge AI systems.

---

# 3. System Architecture

The complete inference pipeline follows a cascaded perception-to-recognition workflow:

```
                 Input Image
                      |
                      v
              YOLO Detector
                      |
                      v
            Meter ROI Extraction
                      |
                      v
       Perspective / Affine Alignment
                      |
                      v
          PaddleOCR Recognition
                      |
                      v
           CTC Sequence Decoding
                      |
                      v
     Structured Digit Output + Confidence
```

---

## Pipeline Description

### 1. Input Stage

Receives raw meter images with arbitrary resolutions and non-standard imaging conditions.

---

### 2. Detector Stage

The YOLO-based detector predicts meter bounding boxes and performs adaptive region extraction.

This stage enables accurate localization of meter regions in complex backgrounds.

---

### 3. Alignment Stage

Optional geometric correction modules perform:

- Perspective transformation
- Affine transformation
- Image rectification

to reduce distortion caused by camera viewpoints.

---

### 4. Recognition Stage

The PaddleOCR-based recognition network extracts sequential visual features and performs digit decoding through CTC-based sequence prediction.

---

### 5. Output Stage

The system outputs:

- Recognized digit sequence
- Confidence score
- Visualization results

for downstream data processing and intelligent monitoring systems.

---

# 4. Directory Structure

```
SmartMeter-Digit-OCR/
│
├── datasets/                 
│   └── Test samples and local validation datasets
│
├── src/                      
│   └── Core source code implementation
│
├── inference_model/          
│   └── Pre-trained model weights
│
├── onnx_deployment/          
│   ├── meter_rec.onnx        
│   ├── ppocr_keys_v1.txt     
│   └── demo_infer.py         
│
├── requirements.txt          
│
└── README.md                 
```

---

# 5. Model Weight Preparation

To maintain a lightweight repository and follow modern open-source engineering practices, large-scale pre-trained weights are separated from the source code.

## Download and Configuration

### Step 1

Download:

```
inference_model_safebox.zip
```

from the repository root directory.

---

### Step 2

Extract the archive and rename the extracted folder:

```
inference_model
```

---

### Step 3

Place the folder under the project root directory:

```
SmartMeter-Digit-OCR/
│
├── src/
├── inference_model/
├── datasets/
├── onnx_deployment/
└── README.md
```

---

# 6. Quick Start

## Step 1: Environment Setup

A clean Python environment is recommended to avoid dependency conflicts.

```bash
git clone https://github.com/wyyyz-web/SmartMeter-Digit-OCR.git

cd SmartMeter-Digit-OCR

pip install -r requirements.txt
```

---

## Step 2: End-to-End Inference

After completing environment configuration and model preparation, run:

```bash
python src/pipeline.py \
    --image_dir ./datasets/test_images/ \
    --det_model_dir ./inference_model/det_model/ \
    --rec_model_dir ./inference_model/rec_model/ \
    --vis_output ./output/
```

---

After execution, visualization results containing:

- Meter bounding boxes
- Recognized digit sequences
- Confidence scores

will automatically be saved into:

```
./output/
```

---

# 7. Industrial-Grade Model Deployment

To support practical edge computing scenarios, the recognition module has been independently optimized for lightweight industrial deployment.

---

## Dynamic Tensor Adaptation

The recognition model has been converted from the original deep learning framework into an ONNX static graph.

Supported dynamic input shape:

```text
[-1, 3, 48, -1]
```

This enables flexible inference for meter images with different widths and digit layouts.

---

## High Reliability Performance

The deployed ONNX recognition model achieves:

```text
Recognition Accuracy: 99.64%

Evaluation Dataset:
5,000 Real-World Blind Test Images
```

under complex industrial imaging conditions.

The results demonstrate strong robustness and practical usability in real deployment scenarios.

---

## Framework-Independent Deployment

The ONNX version removes dependency on the original training framework.

Enter:

```bash
cd onnx_deployment/
```

and run:

```bash
python demo_infer.py
```

Only the lightweight runtime dependency is required:

```text
onnxruntime
```

The deployment architecture enables seamless integration with:

- C++ industrial gateways
- Edge AI devices
- Automated inspection systems
- Smart infrastructure platforms

---

# 8. Project Highlights

SmartMeter-Digit-OCR demonstrates a complete industrial AI workflow:

```
Raw Image Acquisition
          |
          v
Visual Perception
          |
          v
Meter Localization
          |
          v
Digit Sequence Recognition
          |
          v
Structured Data Generation
          |
          v
Industrial Edge Deployment
```

This project bridges the gap between **deep learning research prototypes and real-world intelligent infrastructure applications**.

By combining robust visual perception, sequence recognition, and deployment-oriented optimization, SmartMeter-Digit-OCR provides a scalable solution for:

- Automated meter reading
- Industrial inspection
- Intelligent data acquisition
- Digital infrastructure monitoring

---

# License

This project is released under the MIT License.
