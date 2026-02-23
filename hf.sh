# === 完整设置脚本 ===

# 1. 定义变量
HF_CACHE=~/.cache/huggingface/hub
MODEL_DIR="$HF_CACHE/models--EvolutionaryScale--esm3-sm-open-v1"
SNAPSHOT_DIR="$MODEL_DIR/snapshots/main"
WEIGHTS_SOURCE=/mnt/disk3/tio_nekton4/esm3/weights

echo "创建HuggingFace缓存结构..."

# 2. 创建目录结构
mkdir -p "$SNAPSHOT_DIR/data/weights"
mkdir -p "$MODEL_DIR/refs"

# 3. 创建符号链接到权重文件
echo "链接权重文件..."
ln -sf "$WEIGHTS_SOURCE/esm3_sm_open_v1.pth" \
       "$SNAPSHOT_DIR/data/weights/esm3_sm_open_v1.pth"

ln -sf "$WEIGHTS_SOURCE/esm3_structure_encoder_v0.pth" \
       "$SNAPSHOT_DIR/data/weights/esm3_structure_encoder_v0.pth"

ln -sf "$WEIGHTS_SOURCE/esm3_structure_decoder_v0.pth" \
       "$SNAPSHOT_DIR/data/weights/esm3_structure_decoder_v0.pth"

ln -sf "$WEIGHTS_SOURCE/esm3_function_decoder_v0.pth" \
       "$SNAPSHOT_DIR/data/weights/esm3_function_decoder_v0.pth"

# 4. 查找并链接其他数据文件（tokenizer等）
echo "查找tokenizer数据文件..."
# 查找可能的数据文件
DATA_DIR=/mnt/disk3/tio_nekton4/esm3/esm/data

if [ -d "$DATA_DIR" ]; then
    echo "找到ESM data目录: $DATA_DIR"
    # 链接整个data目录（除了weights）
    for item in "$DATA_DIR"/*; do
        basename=$(basename "$item")
        if [ "$basename" != "weights" ]; then
            ln -sf "$item" "$SNAPSHOT_DIR/data/$basename"
        fi
    done
fi

# 5. 创建refs指向main
echo "main" > "$MODEL_DIR/refs/main"

# 6. 创建.no_exist文件（告诉HF不要再尝试下载）
touch "$MODEL_DIR/.no_exist"

# 7. 验证结构
echo ""
echo "=== 验证创建的结构 ==="
echo "目录: $MODEL_DIR"
echo ""
echo "Refs:"
cat "$MODEL_DIR/refs/main"
echo ""
echo "Snapshot内容:"
find "$SNAPSHOT_DIR" -type f -o -type l | head -20
echo ""
echo "权重文件检查:"
ls -lh "$SNAPSHOT_DIR/data/weights/"

# 8. 测试访问
echo ""
echo "=== 测试HuggingFace缓存访问 ==="
python << 'EOFPY'
import sys
sys.path.insert(0, '/mnt/disk3/tio_nekton4/esm3/esm')
import os
os.environ['HF_HUB_OFFLINE'] = '1'

from huggingface_hub import snapshot_download
from pathlib import Path

try:
    path = Path(snapshot_download(
        repo_id="EvolutionaryScale/esm3-sm-open-v1",
        local_files_only=True
    ))
    print(f"✓ 找到缓存路径: {path}")
    
    # 检查关键文件
    weight_file = path / "data/weights/esm3_sm_open_v1.pth"
    print(f"✓ 主权重文件: {weight_file.exists()}")
    
    # 尝试加载
    import torch
    checkpoint = torch.load(str(weight_file), map_location='cpu')
    print(f"✓ 权重文件可加载")
    print(f"  权重大小: {len(checkpoint)} 个键")
    
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()