#  SmartMeter-Digit-OCR: Robust End-to-End Meter Reading System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)
![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.4%2B-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **"From Pixels to Digits: Bridging the gap between complex industrial environments and reliable data governance."**

---

##  1. 项目简介 (Introduction)

本项目旨在解决复杂工业与自然场景下，智能电表读数的自动化、高精度提取问题。传统的仪表识别方法极易受到光照突变、拍摄视角倾斜、表盘反光及环境噪声的干扰。

为此，本项目设计并实现了一个 **级联双阶视觉架构 (Cascaded Dual-Stage Visual Architecture)**，实现了从原始图像输入到结构化数字输出的端到端 (End-to-End) 推理流水线。本项目不仅聚焦于理论模型的构建，更注重 **MLOps** 视角的工程化部署与数据流转，确保模型在实际场景中的鲁棒性。

---

##  2. 核心特性 (Key Features)

- **Coarse-to-fine 级联检测**：引入基于 YOLO 架构的前置目标检测器，在复杂背景中实现电表 ROI (Region of Interest) 的高精度裁剪与对齐。
- **序列特征解码**：结合 PaddleOCR 的 CRNN-based 网络，利用频率-空间融合策略，对非规则表盘数字序列进行精准解码。
- **工程化极简部署**：彻底分离代码逻辑与大体积权重文件，提供标准化 `requirements.txt` 与一键运行脚本，极大降低二次开发与复现门槛。
- **工业级轻量化部署 (Industrial ONNX Deployment)**：识别模块已成功导出为独立于深度学习框架的 ONNX 静态图，支持动态 Tensor 宽度自适应 (`[-1, 3, 48, -1]`)。在 5,000 张实网盲测数据集上实现了 99.64% 的纯净准确率，具备直接无缝集成至 C++ 边缘计算网关的能力。
---

##  3. 系统架构 (System Architecture)

系统的核心推理数据流如下：

1. **Input Stage**：接收非标准化尺寸的原始图像。
2. **Detector (YOLO)**：预测电表表盘的 Bounding Box，完成目标区域的自适应裁剪。
3. **Alignment**：对倾斜的表盘区域进行仿射变换与透视校正（可选模块）。
4. **Recognizer (PaddleOCR)**：对对齐后的文本行进行特征提取与 CTC 解码。
5. **Output Stage**：输出结构化字符串序列及置信度评分。

---

##  4. 目录结构 (Directory Structure)

```text
SmartMeter-Digit-OCR/
├── datasets/                 # 测试样例与本地验证集
├── src/                      # 核心源代码目录
├── inference_model/          # 预训练模型权重 (需手动解压放置)
├── onnx_deployment/          # 工业级静态图部署模块 (独立运行)
│   ├── meter_rec.onnx        # 支持动态宽度的识别推理模型
│   ├── ppocr_keys_v1.txt     # 字符解码字典
│   └── demo_infer.py         # ONNX Runtime 极简接入示例
├── requirements.txt          # 运行环境依赖清单
└── README.md                 # 项目说明文档
```

---

##  5. 模型权重准备 (Model Weights)

为了保持开源代码库的轻量化并遵循最佳工程实践，预训练模型权重已与代码分离打包。

获取与配置方式：

1. 请下载本仓库根目录下的 `inference_model_safebox.zip` 文件。
2. 将该压缩包解压，并确保解压后的文件夹命名为 `inference_model`。
3. 将该文件夹放置于项目根目录下，使其与 `src` 文件夹同级（如上文树状图所示）。

---

##  6. 快速启动 (Quick Start)

### 步骤 1：环境配置

推荐使用 Conda 构建纯净的虚拟环境，以避免底层依赖冲突：

```bash
git clone https://github.com/wyyyz-web/SmartMeter-Digit-OCR.git

cd SmartMeter-Digit-OCR

pip install -r requirements.txt
```

### 步骤 2：端到端推理验证

系统配置完毕后，通过执行主干流水线脚本，即可对样例图片进行自动化识别：

```bash
python src/pipeline.py \
    --image_dir ./datasets/test_images/ \
    --det_model_dir ./inference_model/det_model/ \
    --rec_model_dir ./inference_model/rec_model/ \
    --vis_output ./output/
```

执行完毕后，带有 Bounding Box 标注与识别结果的可视化图像将自动保存在 `./output/` 文件夹中。

---
---

---
## 7. 工业级模型部署 (Industrial Deployment)

为了适应边缘终端，本项目的识别模块已实现独立轻量化部署。
- **动态张量支持**：模型脱离原生框架，导出为 ONNX 静态图，支持动态宽度自适应 (`[-1, 3, 48, -1]`)。
- **性能指标**：在 5,000 张真实复杂场景盲测集中，实现 99.64% 的识别准确率。
- **一键接入**：进入 `onnx_deployment/` 目录运行 `demo_infer.py`，仅需底层依赖 `onnxruntime` 即可完成毫秒级推理，无缝衔接 C++ 生产环境。
