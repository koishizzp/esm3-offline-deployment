#!/bin/bash
# ESM3 embedding pipeline 快捷运行

INPUT=${1:-CEF.rmdup.faa}
OUTPUT=${2:-./esm3_CEF_operon_rmdup_embeddings}

python get_embeddings_offline.py "$INPUT" -o "$OUTPUT" --half --format both --pooling mean

echo ""
echo "完成！结果在: $OUTPUT"
echo "  - 向量目录: $OUTPUT/embeddings"
echo "  - 元数据:   $OUTPUT/metadata.csv"
echo "  - 汇总:     $OUTPUT/run_summary.json"
