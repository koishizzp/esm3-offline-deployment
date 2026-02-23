#!/usr/bin/env python
# coding=utf-8
"""
Author: Han Wang wanghan0501@foxmail.com
Date: 2025-03-23 20:25:42
LastEditors: Han Wang wanghan0501@foxmail.com
LastEditTime: 2025-03-24 20:14:16
Description: 

Copyright (c) 2025 by StoneWise, All Rights Reserved. 
"""
import gc
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.tokenization import get_esm3_model_tokenizers
import numpy as np
import pandas as pd
import pyfastx
import torch
import gzip
from os.path import join, dirname, basename
from os import makedirs
import pickle
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig, LogitsOutput
from tqdm.auto import tqdm

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
# login()
import torch
from pathlib import Path
from esm.models.esm3 import ESM3
from esm.tokenization import get_esm3_model_tokenizers
import esm.utils.constants.esm3 as esm3_constants

# 导入必要的子模块构造器
from esm.layers.structure_2d import ESM3StructureEncoder, ESM3StructureDecoder
from esm.layers.function_head import ESM3FunctionDecoder # 注意：如果导入失败，请看下方的“提示”

# ================= 1. 离线路径配置 =================
LOCAL_DATA_DIR = "/mnt/disk3/tio_nekton4/esm3-sm-open-v1/Data"
LOCAL_MODEL_PATH = "/mnt/disk3/tio_nekton4/esm3/weights/esm3_sm_open_v1.pth"

def mocked_data_root(model_id: str = "esm3"):
    return Path(LOCAL_DATA_DIR)
esm3_constants.data_root = mocked_data_root

print("🚀 开始手动装配 ESM3 离线模型...")

# ================= 2. 初始化 Tokenizer =================
tokenizers = get_esm3_model_tokenizers("esm3_sm_open_v1")

# ================= 3. 手动构建子模块 (针对 Small Open 版本) =================
# ESM3 Small Open (1.4B) 的标准参数：
d_model = 1536
n_heads = 24
n_layers = 28
v_heads = 1  # 报错信息提示缺这个

# 根据 ESM3 架构初始化结构和功能头
print("🏗️ 正在初始化结构编码器与解码器...")
struct_encoder = ESM3StructureEncoder(d_model=d_model, n_heads=n_heads, n_layers=15)
struct_decoder = ESM3StructureDecoder(d_model=d_model, n_heads=n_heads, n_layers=15)

print("🏗️ 正在初始化功能解码器...")
# 对于 Small Open，功能头通常是一个简单的线性投影或单层 Transformer
func_decoder = ESM3FunctionDecoder(d_model=d_model) 

# ================= 4. 正式实例化 ESM3 =================
# 严格按照报错要求的顺序传入参数
model = ESM3(
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    v_heads=v_heads,
    structure_encoder_fn=struct_encoder,
    structure_decoder_fn=struct_decoder,
    function_decoder_fn=func_decoder,
    tokenizers=tokenizers
)

# ================= 5. 加载权重 =================
print(f"💾 正在加载本地权重: {LOCAL_MODEL_PATH}")
state_dict = torch.load(LOCAL_MODEL_PATH, map_location="cpu")
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
    
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)

model = model.to("cuda").eval()
print("🎉 [大功告成] ESM3 全模块离线加载成功！")
# ================= 离线加载代码 START =================
print("正在加载本地模型...")

# 1. 你的权重文件路径
local_model_path = "/mnt/disk3/tio_nekton4/esm3/weights/esm3_sm_open_v1.pth"

# 2. 初始化 Tokenizer
# 注意：这一步通常需要读取一些词表文件。
# 如果你之前的环境下载过，它会用缓存；如果是全新的纯离线环境，这一步可能还会报错（见文末说明）。
tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)

# 3. 构建模型结构（空壳）
# ESM3_OPEN_SMALL 包含了 sm_open_v1 的所有参数配置
try:
    # 较新版本的 pydantic 使用 model_dump()
    model_config = ESM3_OPEN_SMALL.model_dump()
except AttributeError:
    # 旧版本使用 dict()
    model_config = ESM3_OPEN_SMALL.dict()

model = ESM3(
    tokenizers=tokenizers,
    **model_config
)

# 4. 加载本地 .pth 权重
# map_location='cpu' 防止直接加载到 GPU 爆显存，后面再转
state_dict = torch.load(local_model_path, map_location="cpu")

# 处理权重键值不匹配问题 (比如训练时用了 DDP 会多出 'module.' 前缀)
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

new_state_dict = {}
for k, v in state_dict.items():
    # 移除 'module.' 前缀
    name = k.replace("module.", "")
    new_state_dict[name] = v

# 将权重载入模型
msg = model.load_state_dict(new_state_dict, strict=False)
print(f"权重加载结果: {msg}")

# 转移到 CUDA
model = model.to("cuda")
model.eval() # 极其重要：设置为评估模式
print("本地模型加载成功！")
# ================= 离线加载代码 END =================

EMBEDDING_CONFIG = LogitsConfig(return_embeddings=True)
data_root = "./esm3_CEF_operon_rmdup_embeddings"
makedirs(data_root, exist_ok=True)

def embed_sequence(model: ESM3InferenceClient, sequence: str) -> LogitsOutput:
    # 配置最大允许序列长度（根据模型调整）
    MAX_SEQ_LENGTH = 4096
    
    # 跳过过长序列
    if len(sequence) > MAX_SEQ_LENGTH:
        return None
    
    # 创建蛋白质对象
    protein = ESMProtein(sequence=sequence)
    
    # 禁用梯度计算以节省显存
    with torch.no_grad():
        try:
            # 编码序列
            protein_tensor = model.encode(protein)
            # 获取logits输出
            output = model.logits(protein_tensor, EMBEDDING_CONFIG)
            # 计算平均嵌入
            mean_embedding = torch.mean(output.embeddings, dim=-2).squeeze(0).cpu().numpy()
        except Exception as e:
            # 异常时主动清理显存
            torch.cuda.empty_cache()
            raise e
    return mean_embedding


fq = pyfastx.Fastx('CEF.rmdup.faa')
df = pd.DataFrame(fq, columns=["name", "seq"])

failed_log = []  # 记录失败序列
for row_idx, row in tqdm(df.iterrows(), total=len(df)):
    seq_name = row["name"]
    sequence = row["seq"]
    try:
        # 尝试生成嵌入
        emb = embed_sequence(model, sequence)
        if emb is not None:
            item = (seq_name, sequence, emb)
            # 保存结果和错误日志
            with gzip.open(join(data_root, f"{seq_name}_emb.pkl.gz"), 'wb') as f:
                pickle.dump(item, f)
        else:
            failed_log.append((seq_name, "Skipped: Sequence too long"))
            
    except torch.cuda.OutOfMemoryError:
        # 显存不足处理
        failed_log.append((seq_name, "OOM Error"))
        print(f"⚠️ OOM跳过序列: {seq_name}")
        # 主动清理显存
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        # 其他异常处理
        failed_log.append((seq_name, f"Error: {str(e)}"))
        print(f"⚠️ 异常跳过序列 {seq_name}: {e}")
    
    # 每次迭代后强制清理缓存
    gc.collect()
    torch.cuda.empty_cache()

pd.DataFrame(failed_log, columns=["name", "reason"]).to_csv("failed_sequences.csv", index=False)



