#!/usr/bin/env python
# coding=utf-8
"""
ESM3蛋白质序列嵌入提取工具 - 优化版
特性：
- 批处理优化
- 断点续传
- 内存管理优化
- 详细统计信息
- 多种输出格式
"""
import gc
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import pyfastx
import torch
import gzip
import pickle
from tqdm.auto import tqdm

# ========== 环境配置 ==========
sys.path.insert(0, '/mnt/disk3/tio_nekton4/esm3/esm')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 自动查找snapshot目录
cache_base = Path.home() / ".cache/huggingface/hub/models--EvolutionaryScale--esm3-sm-open-v1/snapshots"
for snapshot_name in ["main", "offline_snapshot_esm3_sm_open_v1"]:
    if (cache_base / snapshot_name).exists():
        LOCAL_CACHE = cache_base / snapshot_name
        break
else:
    snapshots = list(cache_base.glob("*"))
    LOCAL_CACHE = snapshots[0] if snapshots else None

if LOCAL_CACHE is None:
    print("❌ 错误：找不到ESM3缓存目录")
    sys.exit(1)

# Monkey patch
from esm.utils.constants import esm3 as C
@staticmethod
def patched_data_root(model: str):
    return LOCAL_CACHE
C.data_root = patched_data_root

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig


class ESM3EmbeddingExtractor:
    """ESM3嵌入提取器"""
    
    def __init__(
        self,
        model_name: str = 'esm3_sm_open_v1',
        device: str = 'auto',
        max_seq_length: int = 4096,
        batch_size: int = 1,  # ESM3生成模式不支持批处理，但保留接口
        half_precision: bool = False,  # 混合精度
    ):
        """
        初始化提取器
        
        Args:
            model_name: 模型名称
            device: 计算设备 ('cuda', 'cpu', 或 'auto')
            max_seq_length: 最大序列长度
            batch_size: 批处理大小（当前ESM3为1）
            half_precision: 是否使用半精度（节省显存，略微降低精度）
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.half_precision = half_precision
        
        # 自动选择设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.embedding_config = LogitsConfig(return_embeddings=True)
        
        # 统计信息
        self.stats = {
            'processed': 0,
            'skipped_long': 0,
            'failed': 0,
            'total_time': 0,
        }
    
    def load_model(self):
        """加载ESM3模型"""
        if self.model is not None:
            return
        
        print(f"📦 加载ESM3模型: {self.model_name}")
        print(f"   设备: {self.device}")
        print(f"   缓存: {LOCAL_CACHE}")
        
        try:
            self.model = ESM3.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            
            # 半精度优化
            if self.half_precision and self.device == 'cuda':
                self.model = self.model.half()
                print("   ✓ 启用半精度加速")
            
            self.model.eval()
            print("   ✓ 模型加载成功")
            
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            raise
    
    def embed_sequence(self, sequence: str) -> Optional[np.ndarray]:
        """
        提取单个序列的嵌入
        
        Args:
            sequence: 蛋白质序列
            
        Returns:
            嵌入向量或None
        """
        if len(sequence) > self.max_seq_length:
            return None
        
        protein = ESMProtein(sequence=sequence)
        
        with torch.no_grad():
            try:
                # 编码
                protein_tensor = self.model.encode(protein)
                
                # 获取嵌入
                output = self.model.logits(protein_tensor, self.embedding_config)
                
                # 平均池化
                embedding = torch.mean(output.embeddings, dim=-2).squeeze(0)
                
                # 转为numpy
                return embedding.cpu().float().numpy()
                
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return None
            except Exception as e:
                print(f"      ⚠️ 嵌入提取失败: {e}")
                return None
    
    def process_file(
        self,
        input_file: str,
        output_dir: str,
        resume: bool = True,
        save_format: str = 'pkl.gz',
        cleanup_freq: int = 10,
    ) -> dict:
        """
        处理FASTA文件
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            resume: 是否断点续传
            save_format: 保存格式 ('pkl.gz', 'npy', 'both')
            cleanup_freq: 清理频率（每N个序列）
            
        Returns:
            统计信息字典
        """
        self.load_model()
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 读取FASTA
        print(f"\n📖 读取FASTA: {input_file}")
        try:
            fq = pyfastx.Fastx(input_file)
            sequences = list(fq)
            total = len(sequences)
            print(f"   ✓ 读取 {total} 条序列")
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
            return self.stats
        
        # 初始化日志
        failed_log = []
        
        # 进度条
        pbar = tqdm(sequences, desc="🧬 提取嵌入", unit="seq")
        
        start_time = time.time()
        
        for seq_name, sequence in pbar:
            seq_len = len(sequence)
            
            # 检查是否已存在（断点续传）
            output_file = output_path / f"{seq_name}_emb.pkl.gz"
            if resume and output_file.exists():
                self.stats['processed'] += 1
                pbar.set_postfix({
                    '✓': self.stats['processed'],
                    '↷': self.stats['skipped_long'],
                    '✗': self.stats['failed']
                })
                continue
            
            # 检查长度
            if seq_len > self.max_seq_length:
                self.stats['skipped_long'] += 1
                failed_log.append((seq_name, f"过长 ({seq_len}>{self.max_seq_length})"))
                pbar.set_postfix({
                    '✓': self.stats['processed'],
                    '↷': self.stats['skipped_long'],
                    '✗': self.stats['failed']
                })
                continue
            
            # 提取嵌入
            try:
                embedding = self.embed_sequence(sequence)
                
                if embedding is not None:
                    # 保存
                    self._save_embedding(
                        seq_name, sequence, embedding,
                        output_path, save_format
                    )
                    self.stats['processed'] += 1
                else:
                    self.stats['failed'] += 1
                    failed_log.append((seq_name, "嵌入提取返回None"))
                
            except torch.cuda.OutOfMemoryError:
                self.stats['failed'] += 1
                failed_log.append((seq_name, "OOM"))
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                self.stats['failed'] += 1
                failed_log.append((seq_name, str(e)))
            
            # 更新进度条
            pbar.set_postfix({
                '✓': self.stats['processed'],
                '↷': self.stats['skipped_long'],
                '✗': self.stats['failed']
            })
            
            # 定期清理
            if (self.stats['processed'] + self.stats['failed']) % cleanup_freq == 0:
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        self.stats['total_time'] = time.time() - start_time
        
        # 保存失败日志
        if failed_log:
            log_file = output_path / "failed_sequences.csv"
            pd.DataFrame(failed_log, columns=["name", "reason"]).to_csv(
                log_file, index=False
            )
            print(f"\n📝 失败日志: {log_file}")
        
        return self.stats
    
    def _save_embedding(
        self,
        seq_name: str,
        sequence: str,
        embedding: np.ndarray,
        output_path: Path,
        save_format: str
    ):
        """保存嵌入向量"""
        if save_format in ['pkl.gz', 'both']:
            # Pickle格式（包含序列信息）
            pkl_file = output_path / f"{seq_name}_emb.pkl.gz"
            with gzip.open(pkl_file, 'wb') as f:
                pickle.dump((seq_name, sequence, embedding), f)
        
        if save_format in ['npy', 'both']:
            # NumPy格式（仅嵌入向量）
            npy_file = output_path / f"{seq_name}_emb.npy"
            np.save(npy_file, embedding)
    
    def print_stats(self):
        """打印统计信息"""
        total = sum([
            self.stats['processed'],
            self.stats['skipped_long'],
            self.stats['failed']
        ])
        
        if total == 0:
            return
        
        print("\n" + "=" * 70)
        print("📊 处理统计")
        print("=" * 70)
        print(f"总序列数:     {total:>6}")
        print(f"✓ 成功处理:   {self.stats['processed']:>6}  ({self.stats['processed']/total*100:>5.1f}%)")
        print(f"↷ 跳过(过长): {self.stats['skipped_long']:>6}  ({self.stats['skipped_long']/total*100:>5.1f}%)")
        print(f"✗ 失败:       {self.stats['failed']:>6}  ({self.stats['failed']/total*100:>5.1f}%)")
        
        if self.stats['total_time'] > 0:
            speed = self.stats['processed'] / self.stats['total_time']
            print(f"\n⏱️  总耗时:     {self.stats['total_time']:.1f} 秒")
            print(f"🚀 处理速度:   {speed:.2f} 序列/秒")
        
        print("=" * 70)


def verify_outputs(output_dir: str, format: str = 'pkl.gz'):
    """验证输出文件"""
    print(f"\n🔍 验证输出文件...")
    
    output_path = Path(output_dir)
    
    if format == 'pkl.gz':
        files = list(output_path.glob("*.pkl.gz"))
    elif format == 'npy':
        files = list(output_path.glob("*.npy"))
    else:
        files = list(output_path.glob("*.pkl.gz")) + list(output_path.glob("*.npy"))
    
    if not files:
        print("   ⚠️  未找到输出文件")
        return
    
    print(f"   ✓ 找到 {len(files)} 个文件")
    
    # 检查第一个文件
    sample = files[0]
    try:
        if sample.suffix == '.gz':
            with gzip.open(sample, 'rb') as f:
                seq_name, sequence, embedding = pickle.load(f)
        else:
            embedding = np.load(sample)
            seq_name = sample.stem.replace('_emb', '')
            sequence = "N/A"
        
        print(f"\n   示例: {sample.name}")
        print(f"   序列名: {seq_name}")
        if sequence != "N/A":
            print(f"   序列长度: {len(sequence)}")
        print(f"   嵌入形状: {embedding.shape}")
        print(f"   数据类型: {embedding.dtype}")
        print(f"   ✓ 格式正确")
        
    except Exception as e:
        print(f"   ❌ 文件验证失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='ESM3蛋白质序列嵌入提取工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python %(prog)s input.faa -o output_dir
  
  # 启用半精度加速
  python %(prog)s input.faa -o output_dir --half
  
  # 设置最大长度
  python %(prog)s input.faa -o output_dir --max-length 2048
  
  # 保存为npy格式
  python %(prog)s input.faa -o output_dir --format npy
  
  # 禁用断点续传
  python %(prog)s input.faa -o output_dir --no-resume
        """
    )
    
    parser.add_argument('input', help='输入FASTA文件')
    parser.add_argument('-o', '--output', required=True, help='输出目录')
    parser.add_argument('--max-length', type=int, default=4096, help='最大序列长度 (默认: 4096)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='计算设备')
    parser.add_argument('--half', action='store_true', help='使用半精度（节省显存）')
    parser.add_argument('--format', choices=['pkl.gz', 'npy', 'both'], default='pkl.gz', help='输出格式')
    parser.add_argument('--cleanup-freq', type=int, default=10, help='清理频率（默认: 每10个序列）')
    parser.add_argument('--no-resume', action='store_true', help='禁用断点续传')
    
    args = parser.parse_args()
    
    # 打印配置
    print("=" * 70)
    print("🧬 ESM3蛋白质序列嵌入提取工具")
    print("=" * 70)
    print(f"输入文件:     {args.input}")
    print(f"输出目录:     {args.output}")
    print(f"最大长度:     {args.max_length}")
    print(f"计算设备:     {args.device}")
    print(f"半精度:       {'启用' if args.half else '禁用'}")
    print(f"输出格式:     {args.format}")
    print(f"断点续传:     {'启用' if not args.no_resume else '禁用'}")
    print("=" * 70)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"\n❌ 错误: 输入文件不存在: {args.input}")
        return 1
    
    # 创建提取器
    extractor = ESM3EmbeddingExtractor(
        device=args.device,
        max_seq_length=args.max_length,
        half_precision=args.half,
    )
    
    # 处理文件
    try:
        extractor.process_file(
            args.input,
            args.output,
            resume=not args.no_resume,
            save_format=args.format,
            cleanup_freq=args.cleanup_freq,
        )
        
        # 打印统计
        extractor.print_stats()
        
        # 验证输出
        verify_outputs(args.output, args.format)
        
        print(f"\n✅ 完成！嵌入文件保存在: {args.output}")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        return 130
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 清理
        if extractor.device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())