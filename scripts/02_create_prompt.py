#!/usr/bin/env python3
"""
脚本02: 创建GFP生成的Prompt
根据论文方法构建prompt
"""

import os
import sys
import pickle

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.structure_utils import load_pdb
from utils.esm_wrapper import ESM3Generator


def create_gfp_prompt(template_file):
    """
    创建GFP生成prompt
    
    Args:
        template_file: 模板PDB文件路径
        
    Returns:
        ESMProtein对象作为prompt
    """
    print("=" * 60)
    print("创建GFP Prompt")
    print("=" * 60)
    
    # 1. 加载模板结构
    print("\n[步骤 1/3] 加载模板结构...")
    template_data = load_pdb(template_file)
    template_seq = template_data['sequence']
    template_coords = template_data['coordinates']
    
    print(f"  模板序列长度: {len(template_seq)}")
    print(f"  模板序列: {template_seq[:50]}...")
    
    # 2. 构建序列prompt
    print("\n[步骤 2/3] 构建序列prompt...")
    print(f"  目标长度: {PROTEIN_LENGTH}")
    print(f"  关键残基: {len(KEY_RESIDUES)} 个")
    
    # 创建masked序列
    sequence = ['_'] * PROTEIN_LENGTH
    
    # 填充关键残基
    for pos, aa in KEY_RESIDUES.items():
        if pos < PROTEIN_LENGTH:
            sequence[pos] = aa
            print(f"    位置 {pos+1}: {aa}")
    
    prompt_sequence = ''.join(sequence)
    
    # 3. 构建结构prompt
    print("\n[步骤 3/3] 构建结构prompt...")
    print(f"  结构约束位置: {len(STRUCTURE_POSITIONS)} 个")
    print(f"    范围: {min(STRUCTURE_POSITIONS)+1}-{max(STRUCTURE_POSITIONS)+1}")
    
    # 提取关键位置的坐标
    # 注意：需要确保STRUCTURE_POSITIONS在模板坐标范围内
    valid_positions = [p for p in STRUCTURE_POSITIONS if p < len(template_coords)]
    prompt_coords = template_coords[valid_positions]
    
    print(f"    有效坐标点: {len(prompt_coords)}")
    
    # 4. 创建ESMProtein对象
    print("\n[步骤 4/4] 创建ESMProtein对象...")
    generator = ESM3Generator(MODEL_DIR, MODEL_NAME)
    
    # 注意：这里我们只提供序列，结构信息通过其他方式传递
    # 实际使用时可能需要调整
    prompt = generator.create_protein(sequence=prompt_sequence)
    
    # 保存prompt数据
    prompt_data = {
        'sequence': prompt_sequence,
        'key_residues': KEY_RESIDUES,
        'structure_positions': STRUCTURE_POSITIONS,
        'coordinates': prompt_coords,
        'template_data': template_data
    }
    
    return prompt, prompt_data


def save_prompt(prompt_data, output_file):
    """
    保存prompt数据
    
    Args:
        prompt_data: prompt数据字典
        output_file: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(prompt_data, f)
    
    print(f"✓ Prompt数据已保存: {output_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("脚本02: 创建GFP Prompt")
    print("=" * 60)
    
    # 检查模板文件
    template_file = os.path.join(TEMPLATES_DIR, f"{TEMPLATE_PDB}.pdb")
    
    if not os.path.exists(template_file):
        print(f"\n✗ 错误: 模板文件不存在: {template_file}")
        print("请先运行 01_download_template.py")
        sys.exit(1)
    
    print(f"\n模板文件: {template_file}")
    
    # 创建prompt
    try:
        prompt, prompt_data = create_gfp_prompt(template_file)
        
        # 保存
        output_file = os.path.join(PROMPTS_DIR, "gfp_prompt.pkl")
        save_prompt(prompt_data, output_file)
        
        # 显示摘要
        print("\n" + "=" * 60)
        print("Prompt创建摘要")
        print("=" * 60)
        print(f"序列长度: {PROTEIN_LENGTH}")
        print(f"固定残基数: {len(KEY_RESIDUES)}")
        print(f"结构约束点: {len(STRUCTURE_POSITIONS)}")
        print(f"Mask位置数: {prompt_data['sequence'].count('_')}")
        
        print("\n" + "=" * 60)
        print("✓ Prompt准备完成!")
        print("=" * 60)
        print(f"\n下一步: 运行 03_generate_single.py")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
