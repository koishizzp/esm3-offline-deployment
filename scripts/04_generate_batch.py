#!/usr/bin/env python3
"""
脚本04: 批量生成GFP候选
生成多个候选序列用于筛选
"""

import os
import sys
import pickle
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.esm_wrapper import ESM3Generator
from utils.structure_utils import save_to_fasta


def parse_args():
    parser = argparse.ArgumentParser(description='批量生成GFP候选')
    parser.add_argument('--num-candidates', type=int, default=DEFAULT_NUM_CANDIDATES,
                       help=f'生成候选数量 (默认: {DEFAULT_NUM_CANDIDATES})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'批次大小 (默认: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                       help=f'温度参数 (默认: {TEMPERATURE})')
    parser.add_argument('--vary-temp', action='store_true',
                       help='每个候选使用不同温度')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("脚本04: 批量生成GFP候选")
    print("=" * 60)
    print(f"\n参数:")
    print(f"  候选数量: {args.num_candidates}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  基础温度: {args.temperature}")
    print(f"  温度变化: {'是' if args.vary_temp else '否'}")
    
    # 加载prompt
    prompt_file = os.path.join(PROMPTS_DIR, "gfp_prompt.pkl")
    with open(prompt_file, 'rb') as f:
        prompt_data = pickle.load(f)
    
    # 初始化生成器
    generator = ESM3Generator(MODEL_DIR, MODEL_NAME)
    
    # 生成
    num_batches = (args.num_candidates + args.batch_size - 1) // args.batch_size
    
    all_generated = []
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, args.num_candidates)
        batch_size = batch_end - batch_start
        
        print(f"\n批次 {batch_idx+1}/{num_batches} (候选 {batch_start+1}-{batch_end})")
        
        for i in range(batch_size):
            global_idx = batch_start + i
            
            # 可变温度
            if args.vary_temp:
                temp = args.temperature + (i % 10) * 0.05
            else:
                temp = args.temperature
            
            print(f"\n  生成候选 {global_idx+1}/{args.num_candidates} (T={temp:.2f})...")
            
            try:
                prompt = generator.create_protein(sequence=prompt_data['sequence'])
                generated = generator.chain_of_thought_generation(
                    prompt,
                    structure_steps=NUM_STRUCTURE_STEPS,
                    sequence_steps=NUM_SEQUENCE_STEPS,
                    temperature=temp
                )
                
                # 保存
                output_file = os.path.join(CANDIDATES_DIR, f"batch_{batch_idx}_candidate_{i}.fasta")
                header = f">batch_{batch_idx}_candidate_{i}|temp_{temp:.2f}"
                save_to_fasta(generated.sequence, output_file, header=header)
                
                all_generated.append(generated)
                print(f"    ✓ 完成 (长度={len(generated.sequence)})")
                
            except Exception as e:
                print(f"    ✗ 失败: {e}")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("生成完成!")
    print("=" * 60)
    print(f"总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"成功生成: {len(all_generated)}/{args.num_candidates}")
    print(f"平均耗时: {elapsed_time/len(all_generated):.1f} 秒/候选")
    print(f"\n下一步: 运行 05_evaluate_candidates.py")


if __name__ == "__main__":
    main()
