#!/bin/bash
# 快速启动脚本
# 自动运行整个工作流程（支持非交互模式）

set -euo pipefail

echo "======================================================================"
echo "GFP论文复现 - 自动化工作流程"
echo "======================================================================"
echo ""

# 参数
AUTO_CONTINUE="${AUTO_CONTINUE:-0}"
NUM_CANDIDATES="${NUM_CANDIDATES:-100}"
BATCH_SIZE="${BATCH_SIZE:-10}"

if [[ "${1:-}" == "--full" ]]; then
  AUTO_CONTINUE="1"
fi

# 可选激活 conda 环境
if command -v conda >/dev/null 2>&1; then
  if conda env list | grep -q "esm3_env"; then
      echo "[1/7] 激活conda环境..."
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate esm3_env
      echo "✓ 环境已激活"
  else
      echo "[1/7] ⚠ 未检测到 conda 环境 esm3_env，继续使用当前 Python 环境"
  fi
else
  echo "[1/7] ⚠ 未检测到 conda，继续使用当前 Python 环境"
fi

cd "$(dirname "$0")/scripts"

echo ""
echo "[2/7] 下载GFP模板结构..."
python 01_download_template.py

echo ""
echo "[3/7] 创建生成prompt..."
python 02_create_prompt.py

echo ""
echo "[4/7] 生成测试候选..."
python 03_generate_single.py

echo ""
echo "======================================================================"
if [[ "$AUTO_CONTINUE" == "1" ]]; then
  REPLY="y"
  echo "自动模式: 继续批量生成"
else
  read -r -p "测试成功! 是否继续批量生成? (y/n): " -n 1 REPLY
  echo ""
fi

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "[5/7] 批量生成 ${NUM_CANDIDATES} 个候选..."
    echo "警告: 这可能需要几个小时!"
    python 04_generate_batch.py --num-candidates "${NUM_CANDIDATES}" --batch-size "${BATCH_SIZE}"

    echo ""
    echo "[6/7] 评估所有候选..."
    python 05_evaluate_candidates.py

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
