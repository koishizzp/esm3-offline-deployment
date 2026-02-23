#!/bin/bash
# 自动处理当前目录所有FASTA文件

echo "查找FASTA文件..."

# 查找所有.fasta, .faa, .fa文件
for file in *.fasta *.faa *.fa; do
    # 跳过不存在的（glob失败时）
    [ -e "$file" ] || continue
    
    # 生成输出目录名
    basename=$(basename "$file" | sed 's/\.[^.]*$//')
    output_dir="embeddings_${basename}"
    
    echo ""
    echo "========================================"
    echo "📂 文件: $file"
    echo "📁 输出: $output_dir"
    echo "========================================"
    
    # 运行提取
    python get_embeddings_offline.py "$file" -o "$output_dir" --half
    
    # 检查结果
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
