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


## 5. ESM3 Embedding Pipeline（离线）

仓库提供了升级后的离线嵌入流水线：`scripts/get_embeddings_offline.py`。

特性：
- 离线 snapshot 自动发现（读取 `ESM3_SNAPSHOT_DIR`）
- 断点续跑（基于 `metadata.csv`）
- 输入质控（过长/非法氨基酸过滤）
- OOM 后可选 CPU 回退重试
- 输出实验元数据与运行摘要，便于复现实验

示例：

```bash
python scripts/get_embeddings_offline.py data.faa -o embeddings_data --half --format both --pooling mean --l2-normalize
```

### 输入文件是从哪里读取？

`get_embeddings_offline.py` 的第一个位置参数 `input` 就是 **FASTA 文件路径**，脚本按你传入的路径直接读取：

- 传相对路径：相对于你执行命令时的当前目录。
- 传绝对路径：按绝对路径读取。

常见用法：

```bash
# 在仓库根目录运行（推荐）
python scripts/get_embeddings_offline.py data/my_sequences.faa -o data/my_embeddings

# 如果先 cd 到 scripts 目录，需要改相对路径
cd scripts
python get_embeddings_offline.py ../data/my_sequences.faa -o ../data/my_embeddings

# 也可以直接用绝对路径
python scripts/get_embeddings_offline.py /workspace/esm3-offline-deployment/data/my_sequences.faa -o /workspace/esm3-offline-deployment/data/my_embeddings
```

输出目录结构：
- `embeddings_data/embeddings/`: 各序列 embedding 文件
- `embeddings_data/metadata.csv`: 成功样本元数据
- `embeddings_data/failed_sequences.csv`: 失败/跳过原因
- `embeddings_data/run_summary.json`: 本次运行统计摘要
