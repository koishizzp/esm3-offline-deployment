"""
ESM3模型封装 - 使用本地离线缓存
"""

import os
import sys
from pathlib import Path

# 离线模式
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 减少CUDA内存碎片，缓解长时间批量生成的OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# 添加项目路径
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ESM3_SNAPSHOT_DIR, ESM_SOURCE_PATH


def _inject_esm_source_path() -> None:
    """按需注入本地 ESM 源码路径（仅在用户提供时）。"""
    if ESM_SOURCE_PATH and ESM_SOURCE_PATH not in sys.path:
        sys.path.insert(0, ESM_SOURCE_PATH)


def _find_local_snapshot() -> Path:
    """自动查找可用的 ESM3 离线 snapshot。"""
    cache_base = Path(ESM3_SNAPSHOT_DIR)
    preferred = ["main", "offline_snapshot_esm3_sm_open_v1"]

    for snapshot_name in preferred:
        path = cache_base / snapshot_name
        if path.exists():
            return path

    snapshots = sorted([p for p in cache_base.glob("*") if p.is_dir()]) if cache_base.exists() else []
    if snapshots:
        return snapshots[0]

    raise RuntimeError(
        "找不到ESM3缓存目录。请设置环境变量 ESM3_SNAPSHOT_DIR 或先下载离线模型快照。"
    )


def _patch_data_root(local_data_path: Path) -> None:
    from esm.utils.constants import esm3 as C

    @staticmethod
    def patched_data_root(model: str):
        return local_data_path

    C.data_root = patched_data_root


_inject_esm_source_path()
LOCAL_DATA_PATH = _find_local_snapshot()
_patch_data_root(LOCAL_DATA_PATH)


class ESM3Generator:
    """ESM3生成器封装类"""

    def __init__(self, model_dir=None, model_name="esm3_sm_open_v1"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("初始化ESM3生成器...")
        print(f"  模型名称: {model_name}")
        print(f"  设备: {self.device}")

        self.model = None

    def _ensure_model(self):
        if self.model is None:
            from esm.models.esm3 import ESM3

            print("加载ESM3模型（本地缓存）...")
            self.model = ESM3.from_pretrained(self.model_name)

            if self.device.type == "cuda":
                self.model = self.model.to(self.device)

            print("✓ 模型加载完成")

    def clear_cuda_cache(self):
        """主动清理CUDA缓存，降低长批次任务OOM概率。"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def create_protein(self, sequence=None, coordinates=None):
        from esm.sdk.api import ESMProtein

        return ESMProtein(sequence=sequence, coordinates=coordinates)

    def generate_structure(self, prompt, num_steps=200, temperature=0.7):
        self._ensure_model()
        from esm.sdk.api import GenerationConfig

        config = GenerationConfig(track="structure", num_steps=num_steps, temperature=temperature)

        print(f"生成结构... (steps={num_steps}, temp={temperature})")
        with torch.inference_mode():
            result = self.model.generate(prompt, config)
        print("✓ 结构生成完成")
        return result

    def generate_sequence(self, prompt, num_steps=150, temperature=0.7):
        self._ensure_model()
        from esm.sdk.api import GenerationConfig

        config = GenerationConfig(track="sequence", num_steps=num_steps, temperature=temperature)

        print(f"生成序列... (steps={num_steps}, temp={temperature})")
        with torch.inference_mode():
            result = self.model.generate(prompt, config)
        print("✓ 序列生成完成")
        return result

    def predict_structure(self, sequence):
        self._ensure_model()
        from esm.sdk.api import GenerationConfig

        protein = self.create_protein(sequence=sequence)
        base_steps = max(len(sequence) // 16, 10)

        retry_steps = [base_steps, max(base_steps // 2, 8), 8]
        last_error = None
        for attempt, num_steps in enumerate(retry_steps, start=1):
            config = GenerationConfig(track="structure", schedule="cosine", num_steps=num_steps, temperature=0.0)
            try:
                with torch.inference_mode():
                    return self.model.generate(protein, config)
            except Exception as e:
                last_error = e
                print(f"    结构预测失败 (attempt {attempt}/{len(retry_steps)}, steps={num_steps}): {e}")
                self.clear_cuda_cache()

        raise RuntimeError(f"结构预测在{len(retry_steps)}次尝试后仍失败: {last_error}")

    def chain_of_thought_generation(self, prompt, structure_steps=200, sequence_steps=150, temperature=0.7):
        print("=" * 50)
        print("开始Chain-of-Thought生成")
        print("=" * 50)

        print("\n[步骤 1/2] 生成结构tokens...")
        protein_with_structure = self.generate_structure(prompt, num_steps=structure_steps, temperature=temperature)

        print("\n[步骤 2/2] 基于结构生成序列...")
        final_protein = self.generate_sequence(protein_with_structure, num_steps=sequence_steps, temperature=temperature)

        print("\n" + "=" * 50)
        print("✓ Chain-of-Thought生成完成")
        print(f"  生成序列长度: {len(final_protein.sequence)}")
        print("=" * 50)

        return final_protein


if __name__ == "__main__":
    print("测试ESM3Generator")
    generator = ESM3Generator()
    protein = generator.create_protein(sequence="MKTEST")
    print(f"✓ 创建蛋白质: {protein.sequence}")
