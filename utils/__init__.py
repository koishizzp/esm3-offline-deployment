"""
工具函数包初始化
"""

from .structure_utils import load_pdb, calculate_rmsd
from .evaluation import evaluate_candidate

try:
    from .esm_wrapper import ESM3Generator
except ModuleNotFoundError as e:
    # 允许在未安装 torch 的环境下使用非生成相关工具（如 --help、清理脚本等）
    if e.name != 'torch':
        raise
    ESM3Generator = None

__all__ = [
    'ESM3Generator',
    'load_pdb',
    'calculate_rmsd',
    'evaluate_candidate'
]
