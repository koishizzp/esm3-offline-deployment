#!/bin/bash
# ============================================================
# ESM3嵌入提取Web界面 - 启动脚本
# ============================================================

echo "════════════════════════════════════════════════════════"
echo "🧬 ESM3嵌入提取Web界面"
echo "════════════════════════════════════════════════════════"
echo ""

# 检查Gradio是否安装
if ! python -c "import gradio" 2>/dev/null; then
    echo "📦 正在安装Gradio..."
    pip install gradio -q
    
    if [ $? -eq 0 ]; then
        echo "✓ Gradio安装成功"
    else
        echo "❌ Gradio安装失败"
        echo "请手动安装: pip install gradio"
        exit 1
    fi
fi

# 检查必要文件
if [ ! -f "embedding_web_interface.py" ]; then
    echo "❌ 错误: 找不到 embedding_web_interface.py"
    exit 1
fi

if [ ! -f "get_embeddings_offline.py" ]; then
    echo "⚠️  警告: 找不到 get_embeddings_offline.py"
    echo "请确保嵌入提取脚本在当前目录"
fi

echo ""
echo "🚀 启动Web界面..."
echo ""
echo "访问方式："
echo "  本地访问: http://localhost:7860"
echo "  服务器访问: http://服务器IP:7860"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""
echo "════════════════════════════════════════════════════════"
echo ""

# 启动界面
python embedding_web_interface.py
