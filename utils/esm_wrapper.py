"""
ESM3模型封装 - 使用本地缓存（Monkey Patch版本）
"""

import torch
import sys
import os
from pathlib import Path

# 离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 添加ESM3路径
sys.path.insert(0, '/mnt/disk3/tio_nekton4/esm3/esm')

# 自动查找snapshot目录
cache_base = Path.home() / ".cache/huggingface/hub/models--EvolutionaryScale--esm3-sm-open-v1/snapshots"
for snapshot_name in ["main", "offline_snapshot_esm3_sm_open_v1"]:
    if (cache_base / snapshot_name).exists():
        LOCAL_DATA_PATH = cache_base / snapshot_name
        break
else:
    # 找第一个存在的
    snapshots = list(cache_base.glob("*"))
    LOCAL_DATA_PATH = snapshots[0] if snapshots else None

if LOCAL_DATA_PATH is None:
    raise RuntimeError("找不到ESM3缓存目录")

# Monkey patch data_root函数
from esm.utils.constants import esm3 as C

@staticmethod
def patched_data_root(model: str):
    """使用本地缓存路径而不是从HuggingFace下载"""
    return LOCAL_DATA_PATH

# 替换原函数
C.data_root = patched_data_root


class ESM3Generator:
    """ESM3生成器封装类"""
    
    def __init__(self, model_dir=None, model_name="esm3_sm_open_v1"):
        """初始化ESM3生成器"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"初始化ESM3生成器...")
        print(f"  模型名称: {model_name}")
        print(f"  设备: {self.device}")
        
        self.model = None
        
    def _ensure_model(self):
        """加载ESM3模型"""
        if self.model is None:
            from esm.models.esm3 import ESM3
            
            print("加载ESM3模型（本地缓存）...")
            self.model = ESM3.from_pretrained(self.model_name)
            
            # 移动到指定设备
            if self.device.type == 'cuda':
                self.model = self.model.to(self.device)
            
            print("✓ 模型加载完成")
    
    def create_protein(self, sequence=None, coordinates=None):
        """创建ESMProtein对象"""
        from esm.sdk.api import ESMProtein
        return ESMProtein(sequence=sequence, coordinates=coordinates)
    
    def generate_structure(self, prompt, num_steps=200, temperature=0.7):
        """生成蛋白质结构"""
        self._ensure_model()
        from esm.sdk.api import GenerationConfig
        
        config = GenerationConfig(
            track="structure",
            num_steps=num_steps,
            temperature=temperature
        )
        
        print(f"生成结构... (steps={num_steps}, temp={temperature})")
        result = self.model.generate(prompt, config)
        print("✓ 结构生成完成")
        return result
    
    def generate_sequence(self, prompt, num_steps=150, temperature=0.7):
        """生成蛋白质序列"""
        self._ensure_model()
        from esm.sdk.api import GenerationConfig
        
        config = GenerationConfig(
            track="sequence",
            num_steps=num_steps,
            temperature=temperature
        )
        
        print(f"生成序列... (steps={num_steps}, temp={temperature})")
        result = self.model.generate(prompt, config)
        print("✓ 序列生成完成")
        return result
    
    def predict_structure(self, sequence):
        """
        预测蛋白质结构（用于评估）
        
        Args:
            sequence: 蛋白质序列字符串
            
        Returns:
            包含预测结构的ESMProtein对象
        """
        self._ensure_model()
        from esm.sdk.api import GenerationConfig
        
        # 创建蛋白质对象
        protein = self.create_protein(sequence=sequence)
        
        # 使用结构预测（folding）
        num_steps = len(sequence) // 16  # 论文中的启发式
        num_steps = max(num_steps, 10)   # 至少10步
        
        config = GenerationConfig(
            track="structure",
            schedule="cosine",
            num_steps=num_steps,
            temperature=0.0  # 确定性预测
        )
        
        try:
            prediction = self.model.generate(protein, config)
            return prediction
        except Exception as e:
            print(f"    结构预测失败: {e}")
            # 返回原始蛋白质对象（不带结构）
            return protein
    
    def chain_of_thought_generation(
        self, 
        prompt, 
        structure_steps=200, 
        sequence_steps=150,
        temperature=0.7
    ):
        """Chain-of-thought生成（论文方法）"""
        print("=" * 50)
        print("开始Chain-of-Thought生成")
        print("=" * 50)
        
        print("\n[步骤 1/2] 生成结构tokens...")
        protein_with_structure = self.generate_structure(
            prompt, 
            num_steps=structure_steps,
            temperature=temperature
        )
        
        print("\n[步骤 2/2] 基于结构生成序列...")
        final_protein = self.generate_sequence(
            protein_with_structure,
            num_steps=sequence_steps,
            temperature=temperature
        )
        
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
