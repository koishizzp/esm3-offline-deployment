#!/usr/bin/env python3
"""
脚本03: 生成单个GFP候选
快速测试生成流程
"""

import os
import sys
import pickle
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.esm_wrapper import ESM3Generator
from utils.structure_utils import save_to_fasta


def load_prompt(prompt_file):
    """加载prompt数据"""
    with open(prompt_file, 'rb') as f:
        return pickle.load(f)


def generate_single_candidate(generator, prompt_data, temperature=None):
    """
    生成单个候选
    
    Args:
        generator: ESM3Generator实例
        prompt_data: prompt数据
        temperature: 温度参数（可选）
        
    Returns:
        生成的蛋白质对象
    """
    if temperature is None:
        temperature = TEMPERATURE
    
    print("=" * 60)
    print("开始生成GFP候选")
    print("=" * 60)
    
    print(f"\n生成参数:")
    print(f"  温度: {temperature}")
    print(f"  结构生成步数: {NUM_STRUCTURE_STEPS}")
    print(f"  序列生成步数: {NUM_SEQUENCE_STEPS}")
    
    # 创建prompt对象
    prompt = generator.create_protein(sequence=prompt_data['sequence'])
    
    # Chain-of-thought生成
    start_time = time.time()
    
    generated = generator.chain_of_thought_generation(
        prompt,
        structure_steps=NUM_STRUCTURE_STEPS,
        sequence_steps=NUM_SEQUENCE_STEPS,
        temperature=temperature
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n生成耗时: {elapsed_time:.1f} 秒")
    
    return generated


def save_candidate(generated, output_file, index=0):
    """
    保存候选序列
    
    Args:
        generated: 生成的蛋白质对象
        output_file: 输出文件路径
        index: 候选编号
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    header = f">candidate_{index}|length_{len(generated.sequence)}"
    
    save_to_fasta(generated.sequence, output_file, header=header)
    
    # 同时保存完整对象
    pkl_file = output_file.replace('.fasta', '.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(generated, f)
    
    print(f"✓ 完整数据已保存: {pkl_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("脚本03: 生成单个GFP候选")
    print("=" * 60)
    
    # 检查prompt文件
    prompt_file = os.path.join(PROMPTS_DIR, "gfp_prompt.pkl")
    
    if not os.path.exists(prompt_file):
        print(f"\n✗ 错误: Prompt文件不存在: {prompt_file}")
        print("请先运行 02_create_prompt.py")
        sys.exit(1)
    
    print(f"\nPrompt文件: {prompt_file}")
    
    # 加载prompt
    print("\n加载prompt数据...")
    prompt_data = load_prompt(prompt_file)
    print(f"✓ Prompt加载完成")
    
    # 初始化生成器
    print(f"\n初始化ESM3生成器...")
    print(f"  模型目录: {MODEL_DIR}")
    print(f"  模型名称: {MODEL_NAME}")
    
    generator = ESM3Generator(MODEL_DIR, MODEL_NAME)
    
    # 生成
    try:
        generated = generate_single_candidate(generator, prompt_data)
        
        # 保存
        output_file = os.path.join(CANDIDATES_DIR, "candidate_0.fasta")
        save_candidate(generated, output_file, index=0)
        
        # 显示结果
        print("\n" + "=" * 60)
        print("生成结果")
        print("=" * 60)
        print(f"序列长度: {len(generated.sequence)}")
        print(f"序列前50个氨基酸:")
        print(f"  {generated.sequence[:50]}")
        
        # 检查关键残基
        print(f"\n关键残基检查:")
        for pos, expected_aa in KEY_RESIDUES.items():
            if pos < len(generated.sequence):
                actual_aa = generated.sequence[pos]
                match = "✓" if actual_aa == expected_aa else "✗"
                print(f"  位置 {pos+1}: 期望={expected_aa}, 实际={actual_aa} {match}")
        
        print("\n" + "=" * 60)
        print("✓ 候选生成完成!")
        print("=" * 60)
        print(f"\n输出文件: {output_file}")
        print(f"\n下一步:")
        print(f"  - 批量生成: 运行 04_generate_batch.py")
        print(f"  - 或直接评估: 运行 05_evaluate_candidates.py")
        
    except Exception as e:
        print(f"\n✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
