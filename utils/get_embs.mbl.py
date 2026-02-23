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

# This will download the model weights and instantiate the model on your machine.
model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"

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



