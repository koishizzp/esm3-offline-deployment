"""
评估工具
用于评估生成的蛋白质候选
"""

from collections import Counter
import numpy as np
from .structure_utils import calculate_rmsd, sequence_identity


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _to_float(value):
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
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _max_homopolymer_run(sequence):
    max_run, cur = 1, 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return max_run


def check_sequence_constraints(sequence, fixed_residues=None, max_homopolymer_run=5, max_single_aa_fraction=0.25):
    """Filter 1: 序列合理性检查，返回拒绝原因列表（为空表示通过）。"""
    reasons = []

    if not sequence:
        return ["empty_sequence"]

    if "*" in sequence:
        reasons.append("internal_stop_codon")

    invalid = sorted({aa for aa in sequence if aa not in CANONICAL_AA})
    if invalid:
        reasons.append(f"non_canonical_residue:{''.join(invalid)}")

    if _max_homopolymer_run(sequence) >= max_homopolymer_run:
        reasons.append(f"low_complexity_homopolymer>={max_homopolymer_run}")

    aa_count = Counter(sequence)
    max_fraction = max(v / len(sequence) for v in aa_count.values())
    if max_fraction > max_single_aa_fraction:
        dominant = max(aa_count, key=aa_count.get)
        reasons.append(f"composition_bias:{dominant}>{max_single_aa_fraction:.2f}")

    if fixed_residues:
        for pos, expected_aa in fixed_residues.items():
            if pos >= len(sequence):
                reasons.append(f"fixed_residue_out_of_range:{pos+1}")
                continue
            if sequence[pos] != expected_aa:
                reasons.append(f"fixed_residue_violation:{pos+1}{sequence[pos]}!={expected_aa}")

    return reasons


def evaluate_candidate(
    generated_protein,
    template_data,
    chromophore_positions,
    min_ptm=0.8,
    min_plddt=0.8,
    max_chromophore_rmsd=1.5,
    fixed_residues=None,
):
    results = {
        'sequence': generated_protein.sequence,
        'length': len(generated_protein.sequence),
        'pass': False,
        'rejection_reasons': [],
        'metrics': {
            'sequence_identity': None,
            'ptm': None,
            'plddt': None,
            'chromophore_rmsd': None,
        }
    }

    # Filter 1: sequence sanity + fixed residue hard constraints
    seq_reasons = check_sequence_constraints(generated_protein.sequence, fixed_residues=fixed_residues)
    if seq_reasons:
        results['rejection_reasons'].extend(seq_reasons)
        return results

    template_seq = template_data['sequence']
    identity = sequence_identity(generated_protein.sequence, template_seq)
    results['metrics']['sequence_identity'] = identity

    ptm = _to_float(getattr(generated_protein, 'ptm', None))
    plddt_array = _to_numpy(getattr(generated_protein, 'plddt', None))
    plddt = float(np.mean(plddt_array)) if plddt_array is not None else None
    results['metrics']['ptm'] = ptm
    results['metrics']['plddt'] = plddt

    chromophore_rmsd = None
    if hasattr(generated_protein, 'coordinates') and generated_protein.coordinates is not None:
        try:
            gen_coords = _to_numpy(generated_protein.coordinates)
            template_coords = _to_numpy(template_data['coordinates'])
            if gen_coords is not None and gen_coords.ndim == 3:
                gen_coords = gen_coords[:, 1, :]
            if template_coords is not None and template_coords.ndim == 3:
                template_coords = template_coords[:, 1, :]
            chromophore_rmsd = calculate_rmsd(gen_coords, template_coords, indices=chromophore_positions)
            results['metrics']['chromophore_rmsd'] = chromophore_rmsd
        except Exception as e:
            results['rejection_reasons'].append(f"chromophore_rmsd_error:{e}")

    # Filters 2/3: structural constraints
    if ptm is None:
        results['rejection_reasons'].append("missing_ptm")
    elif ptm < min_ptm:
        results['rejection_reasons'].append(f"ptm_below_threshold:{ptm:.3f}<{min_ptm:.3f}")

    if plddt is None:
        results['rejection_reasons'].append("missing_plddt")
    elif plddt < min_plddt:
        results['rejection_reasons'].append(f"plddt_below_threshold:{plddt:.3f}<{min_plddt:.3f}")

    if chromophore_rmsd is None:
        results['rejection_reasons'].append("missing_chromophore_rmsd")
    elif chromophore_rmsd > max_chromophore_rmsd:
        results['rejection_reasons'].append(
            f"chromophore_rmsd_above_threshold:{chromophore_rmsd:.3f}>{max_chromophore_rmsd:.3f}"
        )

    results['pass'] = len(results['rejection_reasons']) == 0
    return results


def rank_candidates(candidates_results, key='ptm'):
    valid_results = [r for r in candidates_results if key in r['metrics'] and r['metrics'][key] is not None]
    return sorted(valid_results, key=lambda x: x['metrics'][key], reverse=True)


def filter_by_criteria(candidates_results, criteria):
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
    if len(sequences) < 2:
        return 0.0
    identities = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            identities.append(sequence_identity(sequences[i], sequences[j]))
    return np.mean(identities)


def generate_summary_stats(candidates_results):
    summary = {
        'total_candidates': len(candidates_results),
        'passed_candidates': sum(1 for r in candidates_results if r['pass']),
    }

    metrics_values = {}
    for result in candidates_results:
        for metric, value in result['metrics'].items():
            if value is not None:
                metrics_values.setdefault(metric, []).append(value)

    for metric, values in metrics_values.items():
        summary[f'{metric}_mean'] = float(np.mean(values))
        summary[f'{metric}_std'] = float(np.std(values))
        summary[f'{metric}_min'] = float(np.min(values))
        summary[f'{metric}_max'] = float(np.max(values))

    sequences = [r['sequence'] for r in candidates_results]
    summary['sequence_diversity'] = float(1.0 - calculate_diversity(sequences))
    return summary


def print_evaluation_report(result, index=None):
    header = f"候选 #{index}" if index is not None else "候选"
    print(f"\n{'=' * 60}")
    print(f"{header}")
    print(f"{'=' * 60}")
    print(f"序列长度: {result['length']}")
    print(f"序列: {result['sequence'][:50]}..." if len(result['sequence']) > 50 else f"序列: {result['sequence']}")

    print("\n指标:")
    for metric, value in result['metrics'].items():
        if value is not None:
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    if result.get('rejection_reasons'):
        print("拒绝原因:")
        for reason in result['rejection_reasons']:
            print(f"  - {reason}")

    print(f"\n评估结果: {'✓ 通过' if result['pass'] else '✗ 未通过'}")
    print(f"{'=' * 60}")


def save_evaluation_results(results, output_file):
    import csv
    if not results:
        print("没有结果可保存")
        return

    all_metrics = sorted({m for result in results for m in result['metrics'].keys()})
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['index', 'length', 'pass', 'rejection_reasons'] + all_metrics + ['sequence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, result in enumerate(results):
            row = {
                'index': i,
                'length': result['length'],
                'pass': result['pass'],
                'rejection_reasons': ';'.join(result.get('rejection_reasons', [])),
                'sequence': result['sequence']
            }
            for metric in all_metrics:
                row[metric] = result['metrics'].get(metric, None)
            writer.writerow(row)
    print(f"✓ 评估结果已保存: {output_file}")
