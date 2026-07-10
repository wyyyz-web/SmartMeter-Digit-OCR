import onnxruntime as ort
import numpy as np

print("[智能电表 OCR 识别模块 - ONNX部署版]")

# 1. 加载字典
dict_path = "ppocr_keys_v1.txt"
with open(dict_path, 'r', encoding='utf-8') as f:
    char_list = [line.strip() for line in f.readlines()]
char_list = ['blank'] + char_list + [' '] 

# 2. 加载模型
model_path = "meter_rec.onnx"
try:
    session = ort.InferenceSession(model_path)
    print("[INFO] 模型与字典加载成功。")
    print("[INFO] 输入张量维度要求: [Batch, 3, 48, 动态宽度]")
except Exception as e:
    print("[ERROR] 加载失败，请确保已安装 onnxruntime 环境。错误信息:", e)
    exit()

print("\n[使用说明]")
print("业务调用规范：将裁切后的电表图像 Resize 至高度 48 像素，")
print("执行归一化后输入 session.run()，通过 CTC 解码对照字典输出最终文本。")
