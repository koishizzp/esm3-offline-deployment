"""
配置文件 - GFP论文复现项目
根据你的服务器环境配置
"""

import os

# ==================== 服务器路径配置 ====================
# ESM3模型权重目录
MODEL_DIR = "/mnt/disk3/tio_nekton4/esm3/weights"

# 项目根目录
PROJECT_ROOT = "/mnt/disk3/tio_nekton4/esm3/projects/gfp_reproduction"

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")
CANDIDATES_DIR = os.path.join(DATA_DIR, "candidates")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# ==================== 模型配置 ====================
# 使用的模型名称
MODEL_NAME = "esm3-sm-open-v1"

# ==================== GFP生成配置（论文参数）====================

# 模板PDB ID
TEMPLATE_PDB = "1QY3"

# 关键残基位置（Python索引，从0开始）
# 论文中的位置：Met1, Thr62, Thr65, Tyr66, Gly67, Arg96, Glu222
KEY_RESIDUES = {
    0: 'M',    # Met1
    61: 'T',   # Thr62
    64: 'T',   # Thr65
    65: 'Y',   # Tyr66
    66: 'G',   # Gly67
    95: 'R',   # Arg96 (A96R mutation)
    221: 'E'   # Glu222
}

# 结构约束位置（论文中的58-71, 96, 222）
STRUCTURE_POSITIONS = list(range(57, 71)) + [95, 221]

# 蛋白质总长度
PROTEIN_LENGTH = 229

# ==================== 生成参数 ====================
# 温度参数
TEMPERATURE = 0.7

# 结构生成步数
NUM_STRUCTURE_STEPS = 200

# 序列生成步数
NUM_SEQUENCE_STEPS = 150

# 批量生成参数
DEFAULT_NUM_CANDIDATES = 100
DEFAULT_BATCH_SIZE = 10

# ==================== 评估参数 ====================
# 最小pTM阈值
MIN_PTM = 0.8

# 最小pLDDT阈值
MIN_PLDDT = 0.8

# chromophore位点最大RMSD（Å）
MAX_CHROMOPHORE_RMSD = 1.5

# chromophore关键位置（用于RMSD计算）
CHROMOPHORE_POSITIONS = [61, 64, 65, 66, 95, 221]

# ==================== 日志配置 ====================
LOG_FILE = os.path.join(RESULTS_DIR, "generation.log")
LOG_LEVEL = "INFO"

# ==================== 其他配置 ====================
# 随机种子
RANDOM_SEED = 42

# 是否使用GPU
USE_GPU = True

# 最大序列长度
MAX_SEQ_LENGTH = 512
