# ESM3 Offline GFP Reproduction (2024.07.01.600583.full.pdf)

本仓库用于离线复现文献 `2024.07.01.600583.full.pdf` 中的 GFP 设计流程（模板约束 + 两阶段生成 + 结构评估）。

## 1. 项目目标

- 使用 ESM3 在离线环境中生成 GFP 候选序列。
- 固定关键位点（如 1, 62, 65, 66, 67, 96, 222）并保留结构约束。
- 对候选进行结构预测与评估，输出可排序结果与 top 序列。

## 2. 快速开始

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 配置环境变量（推荐）

```bash
export PROJECT_ROOT="$(pwd)"
export ESM3_MODEL_DIR="/path/to/weights"
export ESM3_SNAPSHOT_DIR="$HOME/.cache/huggingface/hub/models--EvolutionaryScale--esm3-sm-open-v1/snapshots"
# 如果 esm 没有通过 pip 安装，可指定源码路径：
# export ESM_SOURCE_PATH="/path/to/esm"
```

### 2.3 运行完整流程

非交互完整流程：

```bash
AUTO_CONTINUE=1 NUM_CANDIDATES=100 BATCH_SIZE=10 ./run_all.sh --full
```

或分步运行：

```bash
python scripts/01_download_template.py
python scripts/02_create_prompt.py
python scripts/03_generate_single.py
python scripts/04_generate_batch.py --num-candidates 100 --batch-size 10
python scripts/05_evaluate_candidates.py
python scripts/06_analyze_results.py
```

## 3. 输出结果

生成结果位于 `data/results/`：

- `evaluation_results.csv`: 所有候选评估指标
- `analysis_report.txt`: 汇总分析报告
- `top_candidates.fasta`: top 候选序列

候选序列位于 `data/candidates/`。

## 4. 关键优化说明

- 消除硬编码服务器路径，改为环境变量优先。
- 强化离线 snapshot 自动发现逻辑。
- `run_all.sh` 支持非交互自动跑全流程。
- 修复 PDB 三字母氨基酸到一字母序列转换，避免模板序列错误。

