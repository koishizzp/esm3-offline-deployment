#!/bin/bash
# 自动处理当前目录所有FASTA文件（ESM3 embedding pipeline）

echo "查找FASTA文件..."

for file in *.fasta *.faa *.fa; do
    [ -e "$file" ] || continue

    basename=$(basename "$file" | sed "s/\.[^.]*$//")
    output_dir="embeddings_${basename}"

    echo ""
    echo "========================================"
    echo "📂 文件: $file"
    echo "📁 输出: $output_dir"
    echo "========================================"

    python get_embeddings_offline.py "$file" -o "$output_dir" --half --format both --pooling mean

    if [ $? -eq 0 ]; then
        echo "✅ 成功: $file"
    else
        echo "❌ 失败: $file"
    fi
done

echo ""
echo "========================================"
echo "✅ 批处理完成！"
echo "========================================"
