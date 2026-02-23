#!/bin/bash

# 创建项目目录结构
mkdir -p esm3-embedding-tools
cd esm3-embedding-tools

# 创建子目录
mkdir -p {scripts,docs,examples,templates}

# 复制主要脚本
cp /mnt/disk3/tio_nekton4/esm3/projects/gfp_reproduction/scripts/get_embeddings_offline.py scripts/ 2>/dev/null || echo "跳过 get_embeddings_offline.py"
cp /mnt/disk3/tio_nekton4/esm3/projects/gfp_reproduction/scripts/embedding_web_interface.py scripts/ 2>/dev/null || echo "跳过 embedding_web_interface.py"
cp /mnt/disk3/tio_nekton4/esm3/projects/gfp_reproduction/scripts/start_web_interface.sh scripts/ 2>/dev/null || echo "跳过启动脚本"

echo "✓ 项目结构创建完成"
ls -la
