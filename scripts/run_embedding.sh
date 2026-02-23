#!/bin/bash
# 快捷运行脚本

INPUT=${1:-CEF.rmdup.faa}
OUTPUT=${2:-./esm3_CEF_operon_rmdup_embeddings}

python get_embeddings_offline.py "$INPUT" -o "$OUTPUT" --half

echo ""
echo "完成！结果在: $OUTPUT"
