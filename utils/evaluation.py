"""
评估工具
用于评估生成的蛋白质候选
"""

import numpy as np
from .structure_utils import calculate_rmsd, sequence_identity


def _to_float(value):
    """将numpy/torch标量安全转换为Python float。"""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    return float(value)


def _to_numpy(value):
    """将torch/numpy数据统一转换为numpy数组。"""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def evaluate_candidate(
    generated_protein,
    template_data,
    chromophore_positions,
    min_ptm=0.8,
    min_plddt=0.8,
    max_chromophore_rmsd=1.5
):
    """
    评估单个候选蛋白质
    
    Args:
        generated_protein: 生成的蛋白质对象（ESMProtein）
        template_data: 模板数据（包含序列和坐标）
        chromophore_positions: chromophore关键位置
        min_ptm: 最小pTM阈值
        min_plddt: 最小pLDDT阈值
        max_chromophore_rmsd: 最大chromophore RMSD
        
    Returns:
        dict包含评估结果
    """
    results = {
        'sequence': generated_protein.sequence,
        'length': len(generated_protein.sequence),
        'pass': False,
        'metrics': {
            'sequence_identity': None,
            'ptm': None,
            'plddt': None,
            'chromophore_rmsd': None,
        }
    }
    
    # 1. 序列相同性
    template_seq = template_data['sequence']
    identity = sequence_identity(generated_protein.sequence, template_seq)
    results['metrics']['sequence_identity'] = identity
    
    # 2. 结构质量指标（如果可用）
    if hasattr(generated_protein, 'ptm'):
        ptm = _to_float(generated_protein.ptm)
        results['metrics']['ptm'] = ptm
    else:
        ptm = None
    
    if hasattr(generated_protein, 'plddt'):
        plddt_array = _to_numpy(generated_protein.plddt)
        plddt = float(np.mean(plddt_array)) if plddt_array is not None else None
        results['metrics']['plddt'] = plddt
    else:
        plddt = None
    
    # 3. Chromophore位点RMSD（如果坐标可用）
    chromophore_rmsd = None
    if hasattr(generated_protein, 'coordinates') and generated_protein.coordinates is not None:
        try:
            gen_coords = _to_numpy(generated_protein.coordinates)
            template_coords = _to_numpy(template_data['coordinates'])
            
            chromophore_rmsd = calculate_rmsd(
                gen_coords,
                template_coords,
                indices=chromophore_positions
            )
            results['metrics']['chromophore_rmsd'] = chromophore_rmsd
        except Exception as e:
            print(f"  警告: 无法计算chromophore RMSD: {e}")
    
    # 4. 判断是否通过
    pass_criteria = []
    
    if ptm is not None:
        pass_criteria.append(ptm >= min_ptm)
    
    if plddt is not None:
        pass_criteria.append(plddt >= min_plddt)
    
    if chromophore_rmsd is not None:
        pass_criteria.append(chromophore_rmsd <= max_chromophore_rmsd)
    
    if pass_criteria:
        results['pass'] = all(pass_criteria)
    
    return results


def rank_candidates(candidates_results, key='ptm'):
    """
    对候选结果进行排序
    
    Args:
        candidates_results: 候选评估结果列表
        key: 排序依据的指标
        
    Returns:
        排序后的结果
    """
    # 过滤掉没有该指标的候选
    valid_results = [
        r for r in candidates_results 
        if key in r['metrics'] and r['metrics'][key] is not None
    ]
    
    # 按指标降序排序
    sorted_results = sorted(
        valid_results,
        key=lambda x: x['metrics'][key],
        reverse=True
    )
    
    return sorted_results


def filter_by_criteria(candidates_results, criteria):
    """
    根据标准筛选候选
    
    Args:
        candidates_results: 候选评估结果列表
        criteria: 筛选标准字典，例如 {'ptm': 0.8, 'plddt': 0.8}
        
    Returns:
        筛选后的结果
    """
    filtered = []
    
    for result in candidates_results:
        passes = True
        
        for metric, threshold in criteria.items():
            if metric not in result['metrics']:
                passes = False
                break
            
            value = result['metrics'][metric]
            if value is None or value < threshold:
                passes = False
                break
        
        if passes:
            filtered.append(result)
    
    return filtered


def calculate_diversity(sequences):
    """
    计算序列集合的多样性
    
    Args:
        sequences: 序列列表
        
    Returns:
        平均成对序列相同性（越低越多样）
    """
    if len(sequences) < 2:
        return 0.0
    
    identities = []
    
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            identity = sequence_identity(sequences[i], sequences[j])
            identities.append(identity)
    
    return np.mean(identities)


def generate_summary_stats(candidates_results):
    """
    生成候选集合的统计摘要
    
    Args:
        candidates_results: 候选评估结果列表
        
    Returns:
        统计摘要字典
    """
    summary = {
        'total_candidates': len(candidates_results),
        'passed_candidates': sum(1 for r in candidates_results if r['pass']),
    }
    
    # 收集所有指标
    metrics_values = {}
    
    for result in candidates_results:
        for metric, value in result['metrics'].items():
            if value is not None:
                if metric not in metrics_values:
                    metrics_values[metric] = []
                metrics_values[metric].append(value)
    
    # 计算统计量
    for metric, values in metrics_values.items():
        summary[f'{metric}_mean'] = float(np.mean(values))
        summary[f'{metric}_std'] = float(np.std(values))
        summary[f'{metric}_min'] = float(np.min(values))
        summary[f'{metric}_max'] = float(np.max(values))
    
    # 序列多样性
    sequences = [r['sequence'] for r in candidates_results]
    summary['sequence_diversity'] = float(1.0 - calculate_diversity(sequences))
    
    return summary


def print_evaluation_report(result, index=None):
    """
    打印单个候选的评估报告
    
    Args:
        result: 评估结果字典
        index: 候选编号（可选）
    """
    header = f"候选 #{index}" if index is not None else "候选"
    
    print(f"\n{'=' * 60}")
    print(f"{header}")
    print(f"{'=' * 60}")
    
    print(f"序列长度: {result['length']}")
    print(f"序列: {result['sequence'][:50]}..." if len(result['sequence']) > 50 else f"序列: {result['sequence']}")
    
    print(f"\n指标:")
    for metric, value in result['metrics'].items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    status = "✓ 通过" if result['pass'] else "✗ 未通过"
    print(f"\n评估结果: {status}")
    print(f"{'=' * 60}")


def save_evaluation_results(results, output_file):
    """
    保存评估结果到CSV
    
    Args:
        results: 评估结果列表
        output_file: 输出文件路径
    """
    import csv
    
    if not results:
        print("没有结果可保存")
        return
    
    # 收集所有可能的指标
    all_metrics = set()
    for result in results:
        all_metrics.update(result['metrics'].keys())
    
    all_metrics = sorted(all_metrics)
    
    # 写入CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['index', 'length', 'pass'] + all_metrics + ['sequence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for i, result in enumerate(results):
            row = {
                'index': i,
                'length': result['length'],
                'pass': result['pass'],
                'sequence': result['sequence']
            }
            
            for metric in all_metrics:
                row[metric] = result['metrics'].get(metric, None)
            
            writer.writerow(row)
    
    print(f"✓ 评估结果已保存: {output_file}")


if __name__ == "__main__":
    # 测试代码
    print("评估工具测试")
    
    # 模拟数据
    class MockProtein:
        def __init__(self):
            self.sequence = "MKTEST" * 10
            self.ptm = 0.85
            self.plddt = np.array([0.9] * 60)
            self.coordinates = np.random.rand(60, 3)
    
    template = {
        'sequence': "MKTEST" * 10,
        'coordinates': np.random.rand(60, 3)
    }
    
    protein = MockProtein()
    
    result = evaluate_candidate(
        protein,
        template,
        chromophore_positions=[10, 20, 30]
    )
    
    print_evaluation_report(result, index=1)
