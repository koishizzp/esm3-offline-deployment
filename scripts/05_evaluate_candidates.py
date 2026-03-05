#!/usr/bin/env python3
"""
脚本05: 评估候选序列
对生成的所有候选进行评估
"""

import os
import sys
import glob
import pickle
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.structure_utils import load_from_fasta
from utils.evaluation import (
    evaluate_candidate,
    generate_summary_stats,
    save_evaluation_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description='评估候选序列')
    parser.add_argument('--run-id', type=str, help='仅评估指定run-id前缀的候选')
    parser.add_argument('--pattern', type=str, default='*.fasta',
                       help='候选文件匹配模式（默认: *.fasta）')
    parser.add_argument('--output', type=str,
                       help='输出CSV文件名（默认按run-id自动命名）')
    return parser.parse_args()


def _collect_candidate_files(run_id=None, pattern='*.fasta'):
    if run_id:
        pattern = f"{run_id}_*.fasta"
    return sorted(glob.glob(os.path.join(CANDIDATES_DIR, pattern)))


def main():
    args = parse_args()

    print("=" * 60)
    print("脚本05: 评估候选序列")
    print("=" * 60)
    
    # 查找候选文件（可按 run_id / pattern 过滤）
    fasta_files = _collect_candidate_files(run_id=args.run_id, pattern=args.pattern)
    
    if not fasta_files:
        print(f"\n✗ 错误: 在 {CANDIDATES_DIR} 中没有找到匹配的候选文件")
        print(f"  run_id={args.run_id}, pattern={args.pattern}")
        print("请先运行 03_generate_single.py 或 04_generate_batch.py")
        sys.exit(1)

    print(f"\n找到 {len(fasta_files)} 个候选文件")
    if args.run_id:
        print(f"  run_id筛选: {args.run_id}")

    # 加载模板数据
    prompt_file = os.path.join(PROMPTS_DIR, "gfp_prompt.pkl")
    with open(prompt_file, 'rb') as f:
        prompt_data = pickle.load(f)

    template_data = prompt_data['template_data']

    # 初始化生成器（用于结构预测）
    print("\n初始化ESM3...")
    try:
        from utils.esm_wrapper import ESM3Generator
    except ModuleNotFoundError as e:
        if e.name == "torch":
            print("\n✗ 错误: 未找到 PyTorch (torch) 依赖。")
            print("请先在当前环境安装 torch，例如：")
            print("  pip install torch")
            print("或激活包含 torch 的 conda 环境后重试。")
            sys.exit(1)
        raise

    generator = ESM3Generator(MODEL_DIR, MODEL_NAME)

    # 评估所有候选
    all_results = []
    failed_evaluations = 0
    rejection_reason_counter = Counter()
    
    for i, fasta_file in enumerate(fasta_files):
        print(f"\n[{i+1}/{len(fasta_files)}] 评估: {os.path.basename(fasta_file)}")

        sequences = load_from_fasta(fasta_file)
        for header, sequence in sequences:
            print(f"  序列: {sequence[:30]}...")
            print("  预测结构...")

            try:
                predicted = generator.predict_structure(sequence)

                class ProteinForEval:
                    def __init__(self, seq, pred):
                        self.sequence = seq
                        self.coordinates = getattr(pred, 'coordinates', None)
                        self.ptm = getattr(pred, 'ptm', None)
                        self.plddt = getattr(pred, 'plddt', None)

                protein_obj = ProteinForEval(sequence, predicted)

                result = evaluate_candidate(
                    protein_obj,
                    template_data,
                    CHROMOPHORE_POSITIONS,
                    min_ptm=MIN_PTM,
                    min_plddt=MIN_PLDDT,
                    max_chromophore_rmsd=MAX_CHROMOPHORE_RMSD,
                    fixed_residues=KEY_RESIDUES
                )

                result['file'] = os.path.basename(fasta_file)
                result['header'] = header
                all_results.append(result)
                rejection_reason_counter.update(result.get('rejection_reasons', []))

                status = "✓" if result['pass'] else "✗"
                metrics_str = ", ".join([
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in result['metrics'].items()
                    if v is not None
                ])
                print(f"  {status} {metrics_str}")

            except Exception as e:
                failed_evaluations += 1
                print(f"  ✗ 评估失败: {e}")
                generator.clear_cuda_cache()
    
    # 保存结果
    if args.output:
        output_file = args.output if os.path.isabs(args.output) else os.path.join(RESULTS_DIR, args.output)
    elif args.run_id:
        output_file = os.path.join(RESULTS_DIR, f"evaluation_results_{args.run_id}.csv")
    else:
        output_file = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    save_evaluation_results(all_results, output_file)

    summary = generate_summary_stats(all_results)

    print("\n" + "=" * 60)
    print("评估摘要")
    print("=" * 60)
    print(f"总候选数: {summary['total_candidates']}")
    print(f"通过候选数: {summary['passed_candidates']}")
    print(f"失败评估数: {failed_evaluations}")

    if rejection_reason_counter:
        print("主要拒绝原因 (Top 10):")
        for reason, count in rejection_reason_counter.most_common(10):
            print(f"  - {reason}: {count}")

    if summary['total_candidates'] > 0:
        pass_rate = summary['passed_candidates'] / summary['total_candidates'] * 100
        print(f"通过率: {pass_rate:.1f}%")
    else:
        print("通过率: N/A（没有成功完成评估的候选）")
    
    for key, value in summary.items():
        if key.endswith('_mean'):
            metric = key.replace('_mean', '')
            std = summary.get(f'{metric}_std', 0)
            print(f"{metric}: {value:.4f} ± {std:.4f}")

    print(f"\n结果已保存: {output_file}")
    if args.run_id:
        print(f"\n下一步: 运行 06_analyze_results.py --run-id {args.run_id}")
    else:
        print("\n下一步: 运行 06_analyze_results.py")


if __name__ == "__main__":
    main()
