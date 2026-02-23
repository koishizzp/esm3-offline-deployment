#!/usr/bin/env python3
"""
脚本01: 下载GFP模板结构
从PDB下载1QY3结构文件
"""

import os
import sys
import requests

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TEMPLATES_DIR, TEMPLATE_PDB


def download_pdb(pdb_id, output_dir):
    """
    从RCSB PDB下载结构文件
    
    Args:
        pdb_id: PDB ID
        output_dir: 输出目录
        
    Returns:
        下载的文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    # 检查文件是否已存在
    if os.path.exists(output_file):
        print(f"文件已存在: {output_file}")
        return output_file
    
    # 下载URL
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    print(f"正在下载 {pdb_id} 从 RCSB PDB...")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ 下载完成: {output_file}")
        print(f"  文件大小: {len(response.content)} bytes")
        
        return output_file
        
    except requests.exceptions.RequestException as e:
        print(f"✗ 下载失败: {e}")
        print("\n备用方案:")
        print(f"  请手动访问: {url}")
        print(f"  并保存到: {output_file}")
        return None


def verify_pdb_file(pdb_file):
    """
    验证PDB文件
    
    Args:
        pdb_file: PDB文件路径
        
    Returns:
        是否有效
    """
    if not os.path.exists(pdb_file):
        return False
    
    with open(pdb_file, 'r') as f:
        content = f.read()
        
        # 检查是否包含ATOM记录
        if 'ATOM' not in content:
            print(f"✗ 警告: 文件中没有ATOM记录")
            return False
        
        # 统计原子数
        atom_lines = [line for line in content.split('\n') if line.startswith('ATOM')]
        print(f"✓ PDB文件有效")
        print(f"  原子数: {len(atom_lines)}")
        
        return True


def main():
    """主函数"""
    print("=" * 60)
    print("脚本01: 下载GFP模板结构")
    print("=" * 60)
    
    print(f"\nPDB ID: {TEMPLATE_PDB}")
    print(f"输出目录: {TEMPLATES_DIR}")
    
    # 下载
    pdb_file = download_pdb(TEMPLATE_PDB, TEMPLATES_DIR)
    
    if pdb_file:
        # 验证
        if verify_pdb_file(pdb_file):
            print(f"\n{'=' * 60}")
            print("✓ 模板结构准备完成!")
            print(f"{'=' * 60}")
            print(f"\n下一步: 运行 02_create_prompt.py")
        else:
            print("\n✗ PDB文件验证失败，请检查文件")
            sys.exit(1)
    else:
        print("\n✗ 下载失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
