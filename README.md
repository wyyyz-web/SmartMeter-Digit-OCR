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
│   ├── detector.py           # YOLO 检测器接口封装
│   ├── recognizer.py         # PaddleOCR 识别引擎配置
│   ├── pipeline.py           # 串联检测与识别的端到端主程序
│   └── utils.py              # 图像预处理、NMS 及可视化工具
├── inference_model/          # ⚠️ 预训练模型权重 (需手动解压放置)
│   ├── det_model/            # 目标检测权重文件
│   └── rec_model/            # 序列识别权重文件
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


**Focus Areas:** Artificial Intelligence, Computer Vision, Graph Deep Learning (GDL), and MLOps Data Pipelines.

对于模型架构细节、二次开发建议或业务落地部署问题，欢迎通过提交 Issue 进行探讨。
