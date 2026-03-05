#!/usr/bin/env python3
"""
脚本00: 清理历史候选文件
用于避免评估时混入旧 run 的候选序列。
"""

import os
import sys
import fnmatch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CANDIDATES_DIR


def parse_args():
    parser = argparse.ArgumentParser(description='清理候选目录中的历史文件')
    parser.add_argument('--run-id', type=str, help='仅清理指定 run-id 前缀文件（如 run_20260305_083828）')
    parser.add_argument('--all', action='store_true', help='清理候选目录中所有 fasta/pkl 文件')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将删除的文件，不实际删除')
    return parser.parse_args()


def _target_patterns(run_id: str | None, clean_all: bool):
    if clean_all:
        return ['*.fasta', '*.pkl']
    if run_id:
        return [f'{run_id}_*.fasta', f'{run_id}_*.pkl']
    return ['candidate_*.fasta', 'candidate_*.pkl']


def main():
    args = parse_args()

    if not args.all and not args.run_id:
        print('未提供 --run-id，默认只清理单测产物 candidate_*.fasta / candidate_*.pkl')

    patterns = _target_patterns(args.run_id, args.all)

    if not os.path.isdir(CANDIDATES_DIR):
        print(f'候选目录不存在: {CANDIDATES_DIR}')
        return

    all_files = os.listdir(CANDIDATES_DIR)
    targets: list[str] = []
    for name in all_files:
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            path = os.path.join(CANDIDATES_DIR, name)
            if os.path.isfile(path):
                targets.append(path)

    if not targets:
        print('没有找到需要清理的文件。')
        return

    print(f'找到 {len(targets)} 个待清理文件:')
    for path in sorted(targets):
        print(f'  - {os.path.basename(path)}')

    if args.dry_run:
        print('\n[DRY-RUN] 未执行删除。')
        return

    for path in targets:
        os.remove(path)

    print(f'\n✓ 已删除 {len(targets)} 个文件。')


if __name__ == '__main__':
    main()
