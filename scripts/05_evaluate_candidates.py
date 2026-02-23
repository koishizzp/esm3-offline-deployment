#!/usr/bin/env python3
"""
脚本05: 评估候选序列
对生成的所有候选进行评估
"""

import os
import sys
import glob
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.esm_wrapper import ESM3Generator
from utils.structure_utils import load_from_fasta
from utils.evaluation import (
    evaluate_candidate,
    rank_candidates,
    generate_summary_stats,
    save_evaluation_results,
    print_evaluation_report
)


def main():
    print("=" * 60)
    print("脚本05: 评估候选序列")
    print("=" * 60)
    
    # 查找所有候选文件
    fasta_files = glob.glob(os.path.join(CANDIDATES_DIR, "*.fasta"))
    
    if not fasta_files:
        print(f"\n✗ 错误: 在 {CANDIDATES_DIR} 中没有找到候选文件")
        print("请先运行 03_generate_single.py 或 04_generate_batch.py")
        sys.exit(1)
    
    print(f"\n找到 {len(fasta_files)} 个候选文件")
    
    # 加载模板数据
    prompt_file = os.path.join(PROMPTS_DIR, "gfp_prompt.pkl")
    with open(prompt_file, 'rb') as f:
        prompt_data = pickle.load(f)
    
    template_data = prompt_data['template_data']
    
    # 初始化生成器（用于结构预测）
    print("\n初始化ESM3...")
    generator = ESM3Generator(MODEL_DIR, MODEL_NAME)
    
    # 评估所有候选
    all_results = []
    
    for i, fasta_file in enumerate(fasta_files):
        print(f"\n[{i+1}/{len(fasta_files)}] 评估: {os.path.basename(fasta_file)}")
        
        # 加载序列
        sequences = load_from_fasta(fasta_file)
        
        for header, sequence in sequences:
            print(f"  序列: {sequence[:30]}...")
            
            # 结构预测
            print(f"  预测结构...")
            try:
                predicted = generator.predict_structure(sequence)
                
                # 创建评估用的蛋白质对象
                class ProteinForEval:
                    def __init__(self, seq, pred):
                        self.sequence = seq
                        self.coordinates = getattr(pred, 'coordinates', None)
                        self.ptm = getattr(pred, 'ptm', None)
                        self.plddt = getattr(pred, 'plddt', None)
                
                protein_obj = ProteinForEval(sequence, predicted)
                
                # 评估
                result = evaluate_candidate(
                    protein_obj,
                    template_data,
                    CHROMOPHORE_POSITIONS,
                    min_ptm=MIN_PTM,
                    min_plddt=MIN_PLDDT,
                    max_chromophore_rmsd=MAX_CHROMOPHORE_RMSD
                )
                
                result['file'] = os.path.basename(fasta_file)
                result['header'] = header
                
                all_results.append(result)
                
                # 简要报告
                status = "✓" if result['pass'] else "✗"
                metrics_str = ", ".join([
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in result['metrics'].items()
                    if v is not None
                ])
                print(f"  {status} {metrics_str}")
                
            except Exception as e:
                print(f"  ✗ 评估失败: {e}")
    
    # 保存结果
    output_file = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    save_evaluation_results(all_results, output_file)
    
    # 生成统计摘要
    summary = generate_summary_stats(all_results)
    
    print("\n" + "=" * 60)
    print("评估摘要")
    print("=" * 60)
    print(f"总候选数: {summary['total_candidates']}")
    print(f"通过候选数: {summary['passed_candidates']}")
    print(f"通过率: {summary['passed_candidates']/summary['total_candidates']*100:.1f}%")
    
    for key, value in summary.items():
        if key.endswith('_mean'):
            metric = key.replace('_mean', '')
            std = summary.get(f'{metric}_std', 0)
            print(f"{metric}: {value:.4f} ± {std:.4f}")
    
    print(f"\n结果已保存: {output_file}")
    print(f"\n下一步: 运行 06_analyze_results.py")


if __name__ == "__main__":
    main()
