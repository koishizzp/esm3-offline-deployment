#!/usr/bin/env python3
"""
脚本04: 批量生成GFP候选
生成多个候选序列用于筛选
"""

import os
import sys
import json
import pickle
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
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
    parser.add_argument('--run-id', type=str,
                       help='本次运行ID（默认自动生成时间戳）')
    parser.add_argument('--clean', action='store_true',
                       help='开始前清理本run-id下已存在候选文件')
    parser.add_argument('--clean-all', action='store_true',
                       help='开始前清理候选目录中所有fasta/pkl文件')
    return parser.parse_args()


def _cleanup_candidates(run_id=None, clean_all=False):
    patterns = ["*.fasta", "*.pkl"] if clean_all or not run_id else [f"{run_id}_*.fasta", f"{run_id}_*.pkl"]
    removed = 0
    for pattern in patterns:
        for file in [os.path.join(CANDIDATES_DIR, f) for f in os.listdir(CANDIDATES_DIR) if __import__('fnmatch').fnmatch(f, pattern)]:
            if os.path.isfile(file):
                os.remove(file)
                removed += 1
    return removed


def main():
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("脚本04: 批量生成GFP候选")
    print("=" * 60)
    print(f"\n参数:")
    print(f"  运行ID: {run_id}")
    print(f"  候选数量: {args.num_candidates}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  基础温度: {args.temperature}")
    print(f"  温度变化: {'是' if args.vary_temp else '否'}")

    if args.clean_all:
        removed = _cleanup_candidates(clean_all=True)
        print(f"  清理模式: 全量清理 ({removed} 个文件)")
    elif args.clean:
        removed = _cleanup_candidates(run_id=run_id, clean_all=False)
        print(f"  清理模式: 按run_id清理 ({removed} 个文件)")

    # 加载prompt
    prompt_file = os.path.join(PROMPTS_DIR, "gfp_prompt.pkl")
    with open(prompt_file, 'rb') as f:
        prompt_data = pickle.load(f)

    # 初始化生成器
    try:
        from utils.esm_wrapper import ESM3Generator
    except ModuleNotFoundError as e:
        if e.name == "torch":
            print("\n✗ 错误: 未找到 PyTorch (torch) 依赖。")
            print("请先在当前环境安装 torch，或激活包含 torch 的 conda 环境。")
            sys.exit(1)
        raise

    generator = ESM3Generator(MODEL_DIR, MODEL_NAME)

    # 生成
    num_batches = (args.num_candidates + args.batch_size - 1) // args.batch_size
    
    success_count = 0
    failure_count = 0
    generated_files = []
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

                # 保存（包含run_id，避免混入历史文件）
                filename = f"{run_id}_batch_{batch_idx}_candidate_{i}.fasta"
                output_file = os.path.join(CANDIDATES_DIR, filename)
                header = f">{run_id}|batch_{batch_idx}_candidate_{i}|temp_{temp:.2f}"
                save_to_fasta(generated.sequence, output_file, header=header)
                
                success_count += 1
                generated_files.append(filename)
                print(f"    ✓ 完成 (长度={len(generated.sequence)})")

                # 及时释放对象，避免长任务中显存累积
                del generated
                generator.clear_cuda_cache()
                
            except Exception as e:
                failure_count += 1
                print(f"    ✗ 失败: {e}")
                generator.clear_cuda_cache()
    
    elapsed_time = time.time() - start_time

    manifest = {
        "run_id": run_id,
        "num_requested": args.num_candidates,
        "success_count": success_count,
        "failure_count": failure_count,
        "temperature": args.temperature,
        "vary_temp": args.vary_temp,
        "generated_files": generated_files,
    }
    manifest_file = os.path.join(RESULTS_DIR, f"{run_id}_generation_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("生成完成!")
    print("=" * 60)
    print(f"运行ID: {run_id}")
    print(f"总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"成功生成: {success_count}/{args.num_candidates}")

    if success_count > 0:
        print(f"平均耗时: {elapsed_time/success_count:.1f} 秒/候选")
    else:
        print("平均耗时: N/A（没有成功生成的候选）")
    print(f"\n下一步: 运行 05_evaluate_candidates.py")


if __name__ == "__main__":
    main()
