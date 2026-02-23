#!/bin/bash
# 快速启动脚本
# 自动运行整个工作流程

set -e  # 遇到错误立即停止

echo "======================================================================"
echo "GFP论文复现 - 自动化工作流程"
echo "======================================================================"
echo ""

# 检查conda环境
if ! conda env list | grep -q "esm3_env"; then
    echo "错误: conda环境 'esm3_env' 不存在"
    echo "请先部署ESM3"
    exit 1
fi

# 激活环境
echo "[1/7] 激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate esm3_env
echo "✓ 环境已激活"

# 进入脚本目录
cd "$(dirname "$0")/scripts"

# 步骤1: 下载模板
echo ""
echo "[2/7] 下载GFP模板结构..."
python 01_download_template.py

# 步骤2: 创建prompt
echo ""
echo "[3/7] 创建生成prompt..."
python 02_create_prompt.py

# 步骤3: 生成单个测试
echo ""
echo "[4/7] 生成测试候选..."
python 03_generate_single.py

# 询问是否继续批量生成
echo ""
echo "======================================================================"
read -p "测试成功! 是否继续批量生成? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 步骤4: 批量生成
    echo ""
    read -p "输入要生成的候选数量 [默认100]: " num_candidates
    num_candidates=${num_candidates:-100}
    
    echo "[5/7] 批量生成 $num_candidates 个候选..."
    echo "警告: 这可能需要几个小时!"
    python 04_generate_batch.py --num-candidates $num_candidates --batch-size 10
    
    # 步骤5: 评估
    echo ""
    echo "[6/7] 评估所有候选..."
    python 05_evaluate_candidates.py
    
    # 步骤6: 分析
    echo ""
    echo "[7/7] 生成分析报告..."
    python 06_analyze_results.py
    
    echo ""
    echo "======================================================================"
    echo "✓ 全部完成!"
    echo "======================================================================"
    echo ""
    echo "查看结果:"
    echo "  - 分析报告: ../data/results/analysis_report.txt"
    echo "  - Top候选: ../data/results/top_candidates.fasta"
    echo "  - 详细数据: ../data/results/evaluation_results.csv"
else
    echo ""
    echo "======================================================================"
    echo "✓ 测试完成!"
    echo "======================================================================"
    echo ""
    echo "后续步骤:"
    echo "  cd scripts"
    echo "  python 04_generate_batch.py --num-candidates 100"
    echo "  python 05_evaluate_candidates.py"
    echo "  python 06_analyze_results.py"
fi
