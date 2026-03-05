"""
配置文件 - GFP论文复现项目
优先使用环境变量，避免硬编码服务器路径。
"""

import os
from pathlib import Path


# ==================== 路径配置（可通过环境变量覆盖） ====================
_REPO_ROOT = Path(__file__).resolve().parent

# 项目根目录
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", str(_REPO_ROOT))

# ESM3模型权重目录
# 兼容历史变量名 MODEL_DIR 与推荐变量 ESM3_MODEL_DIR
MODEL_DIR = os.environ.get(
    "ESM3_MODEL_DIR",
    os.environ.get("MODEL_DIR", str(Path(PROJECT_ROOT) / "weights")),
)

# 可选：本地 ESM 源码路径（用于未 pip install esm 的场景）
ESM_SOURCE_PATH = os.environ.get("ESM_SOURCE_PATH")

# 可选：离线快照根目录（默认 HuggingFace cache）
ESM3_SNAPSHOT_DIR = os.environ.get(
    "ESM3_SNAPSHOT_DIR",
    str(Path.home() / ".cache/huggingface/hub/models--EvolutionaryScale--esm3-sm-open-v1/snapshots"),
)

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")
CANDIDATES_DIR = os.path.join(DATA_DIR, "candidates")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# ==================== 模型配置 ====================
MODEL_NAME = "esm3-sm-open-v1"

# ==================== GFP生成配置（论文参数）====================
TEMPLATE_PDB = "1QY3"

# 关键残基位置（Python索引，从0开始）
# 论文中的位置：Met1, Thr62, Thr65, Tyr66, Gly67, Arg96, Glu222
KEY_RESIDUES = {
    0: "M",  # Met1
    61: "T",  # Thr62
    64: "T",  # Thr65
    65: "Y",  # Tyr66
    66: "G",  # Gly67
    95: "R",  # Arg96 (A96R mutation)
    221: "E",  # Glu222
}

# 结构约束位置（论文中的58-71, 96, 222）
STRUCTURE_POSITIONS = list(range(57, 71)) + [95, 221]

PROTEIN_LENGTH = 229

# ==================== 生成参数 ====================
TEMPERATURE = 0.7
NUM_STRUCTURE_STEPS = 200
NUM_SEQUENCE_STEPS = 150
DEFAULT_NUM_CANDIDATES = 30
DEFAULT_BATCH_SIZE = 10

# ==================== 评估参数 ====================
MIN_PTM = 0.8
MIN_PLDDT = 0.8
MAX_CHROMOPHORE_RMSD = 1.5
CHROMOPHORE_POSITIONS = [61, 64, 65, 66, 95, 221]

# ==================== 日志配置 ====================
LOG_FILE = os.path.join(RESULTS_DIR, "generation.log")
LOG_LEVEL = "INFO"

# ==================== 其他配置 ====================
RANDOM_SEED = 42
USE_GPU = True
MAX_SEQ_LENGTH = 512
