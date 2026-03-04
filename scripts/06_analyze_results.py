#!/usr/bin/env python3
"""
脚本06: 分析结果
生成最终分析报告和Top候选
"""

import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

EXPECTED_METRICS = ['ptm', 'plddt', 'chromophore_rmsd', 'sequence_identity']


def parse_args():
    parser = argparse.ArgumentParser(description='分析评估结果')
    parser.add_argument('--run-id', type=str, help='指定run-id（读取 evaluation_results_<run-id>.csv）')
    parser.add_argument('--results-file', type=str, help='显式指定评估CSV文件路径')
    parser.add_argument('--top-n', type=int, default=10, help='导出Top候选数量')
    return parser.parse_args()


def resolve_results_file(args):
    if args.results_file:
        return args.results_file if os.path.isabs(args.results_file) else os.path.join(RESULTS_DIR, args.results_file)
    if args.run_id:
        return os.path.join(RESULTS_DIR, f"evaluation_results_{args.run_id}.csv")
    return os.path.join(RESULTS_DIR, "evaluation_results.csv")


def load_evaluation_results(csv_file):
    """加载评估结果"""
    results = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                if key in EXPECTED_METRICS:
                    try:
                        row[key] = float(row[key]) if row[key] else None
                    except Exception:
                        row[key] = None
                elif key in ['index', 'length']:
                    row[key] = int(row[key])
                elif key == 'pass':
                    row[key] = row[key].lower() == 'true'

            for metric in EXPECTED_METRICS:
                row.setdefault(metric, None)

            results.append(row)

    return results


def generate_report(results, output_file, top_n=10):
    """生成文本报告"""

    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GFP候选生成 - 分析报告\n")
        f.write("=" * 70 + "\n\n")

        f.write("总体统计:\n")
        f.write("-" * 70 + "\n")
        f.write(f"总候选数: {len(results)}\n")

        passed = [r for r in results if r['pass']]
        f.write(f"通过候选数: {len(passed)}\n")
        if results:
            f.write(f"通过率: {len(passed)/len(results)*100:.1f}%\n\n")
        else:
            f.write("通过率: N/A（没有可分析的候选）\n\n")

        f.write("指标统计:\n")
        f.write("-" * 70 + "\n")

        for metric in EXPECTED_METRICS:
            values = [r.get(metric) for r in results if r.get(metric) is not None]

            if values:
                import numpy as np
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  平均值: {np.mean(values):.4f}\n")
                f.write(f"  标准差: {np.std(values):.4f}\n")
                f.write(f"  最小值: {np.min(values):.4f}\n")
                f.write(f"  最大值: {np.max(values):.4f}\n")

        f.write("\n\n" + "=" * 70 + "\n")
        f.write(f"Top {top_n} 候选 (按pTM排序)\n")
        f.write("=" * 70 + "\n")

        sorted_results = sorted(
            [r for r in results if r.get('ptm') is not None],
            key=lambda x: x['ptm'],
            reverse=True
        )

        for i, result in enumerate(sorted_results[:top_n]):
            f.write(f"\n排名 {i+1}:\n")
            f.write(f"  索引: {result['index']}\n")
            f.write(f"  pTM: {result['ptm']:.4f}\n")
            f.write(f"  pLDDT: {result.get('plddt', 'N/A')}\n")
            f.write(f"  序列相似度: {result.get('sequence_identity', 'N/A')}\n")
            f.write(f"  Chromophore RMSD: {result.get('chromophore_rmsd', 'N/A')}\n")
            f.write(f"  序列: {result['sequence'][:50]}...\n")

    print(f"✓ 报告已保存: {output_file}")


def save_top_candidates(results, output_file, top_n=10):
    """保存Top候选到FASTA"""

    sorted_results = sorted(
        [r for r in results if r.get('ptm') is not None],
        key=lambda x: x['ptm'],
        reverse=True
    )

    with open(output_file, 'w') as f:
        for i, result in enumerate(sorted_results[:top_n]):
            header = (f">rank_{i+1}|index_{result['index']}|"
                      f"pTM_{result['ptm']:.3f}|"
                      f"identity_{result.get('sequence_identity', 0):.2f}")

            f.write(f"{header}\n")
            seq = result['sequence']
            for j in range(0, len(seq), 80):
                f.write(f"{seq[j:j+80]}\n")

    print(f"✓ Top {top_n} 候选已保存: {output_file}")


def main():
    args = parse_args()

    print("=" * 60)
    print("脚本06: 分析结果")
    print("=" * 60)

    results_file = resolve_results_file(args)

    if not os.path.exists(results_file):
        print(f"\n✗ 错误: 评估结果文件不存在: {results_file}")
        print("请先运行 05_evaluate_candidates.py")
        sys.exit(1)

    print(f"\n加载评估结果: {results_file}")
    results = load_evaluation_results(results_file)
    print(f"✓ 加载了 {len(results)} 个候选的评估结果")

    suffix = f"_{args.run_id}" if args.run_id else ""

    print("\n生成分析报告...")
    report_file = os.path.join(RESULTS_DIR, f"analysis_report{suffix}.txt")
    generate_report(results, report_file, top_n=args.top_n)

    print("\n保存Top候选...")
    top_file = os.path.join(RESULTS_DIR, f"top_candidates{suffix}.fasta")
    save_top_candidates(results, top_file, top_n=args.top_n)

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

    sorted_results = sorted(
        [r for r in results if r.get('ptm') is not None],
        key=lambda x: x['ptm'],
        reverse=True
    )

    print("\nTop 3 候选:")
    for i, r in enumerate(sorted_results[:3]):
        print(f"\n{i+1}. 候选 #{r['index']}")
        print(f"   pTM: {r['ptm']:.4f}")
        print(f"   pLDDT: {r.get('plddt', 'N/A')}")
        print(f"   序列相似度: {r.get('sequence_identity', 'N/A')}")
        print(f"   序列: {r['sequence'][:40]}...")

    print(f"\n完整报告: {report_file}")
    print(f"Top候选: {top_file}")


if __name__ == "__main__":
    main()
