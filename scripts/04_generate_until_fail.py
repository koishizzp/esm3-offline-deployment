#!/usr/bin/env python3
"""
脚本04变种: 连续生成直到首次失败
用途：测量在当前环境下连续生成上限（含显存稳定性）
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
    parser = argparse.ArgumentParser(description='连续生成GFP候选，失败即停止')
    parser.add_argument('--max-candidates', type=int, default=1000,
                        help='最大尝试生成数量（默认: 1000）')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help=f'温度参数 (默认: {TEMPERATURE})')
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
    print("脚本04变种: 连续生成直到首次失败")
    print("=" * 60)
    print(f"\n参数:")
    print(f"  运行ID: {run_id}")
    print(f"  最大尝试数: {args.max_candidates}")
    print(f"  温度: {args.temperature}")

    if args.clean_all:
        removed = _cleanup_candidates(clean_all=True)
        print(f"  清理模式: 全量清理 ({removed} 个文件)")
    elif args.clean:
        removed = _cleanup_candidates(run_id=run_id, clean_all=False)
        print(f"  清理模式: 按run_id清理 ({removed} 个文件)")

    prompt_file = os.path.join(PROMPTS_DIR, "gfp_prompt.pkl")
    with open(prompt_file, 'rb') as f:
        prompt_data = pickle.load(f)

    try:
        from utils.esm_wrapper import ESM3Generator
    except ModuleNotFoundError as e:
        if e.name == "torch":
            print("\n✗ 错误: 未找到 PyTorch (torch) 依赖。")
            print("请先在当前环境安装 torch，或激活包含 torch 的 conda 环境。")
            sys.exit(1)
        raise

    generator = ESM3Generator(MODEL_DIR, MODEL_NAME)

    success_count = 0
    failure_info = None
    generated_files = []
    start_time = time.time()

    for i in range(args.max_candidates):
        print(f"\n生成候选 {i+1}/{args.max_candidates} (T={args.temperature:.2f})...")
        try:
            prompt = generator.create_protein(sequence=prompt_data['sequence'])
            generated = generator.chain_of_thought_generation(
                prompt,
                structure_steps=NUM_STRUCTURE_STEPS,
                sequence_steps=NUM_SEQUENCE_STEPS,
                temperature=args.temperature,
            )

            filename = f"{run_id}_limit_candidate_{i}.fasta"
            output_file = os.path.join(CANDIDATES_DIR, filename)
            header = f">{run_id}|limit_candidate_{i}|temp_{args.temperature:.2f}"
            save_to_fasta(generated.sequence, output_file, header=header)

            success_count += 1
            generated_files.append(filename)
            print(f"  ✓ 完成 (长度={len(generated.sequence)})")

            del generated
            generator.clear_cuda_cache()

        except Exception as e:
            failure_info = {
                "failed_at_index": i,
                "failed_at_candidate": i + 1,
                "error": str(e),
            }
            print(f"  ✗ 失败，停止测试: {e}")
            generator.clear_cuda_cache()
            break

    elapsed_time = time.time() - start_time

    manifest = {
        "run_id": run_id,
        "max_requested": args.max_candidates,
        "success_count_before_failure": success_count,
        "failure": failure_info,
        "temperature": args.temperature,
        "generated_files": generated_files,
        "elapsed_seconds": elapsed_time,
    }

    manifest_file = os.path.join(RESULTS_DIR, f"{run_id}_generation_limit_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("上限测试完成")
    print("=" * 60)
    print(f"运行ID: {run_id}")
    print(f"连续成功生成数量: {success_count}")
    if failure_info:
        print(f"首次失败位置: 第 {failure_info['failed_at_candidate']} 个")
        print(f"失败原因: {failure_info['error']}")
    else:
        print("在最大尝试数内未失败")
    print(f"总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"结果清单: {manifest_file}")


if __name__ == "__main__":
    main()
