"""
结构工具
用于处理蛋白质结构相关操作
"""

import numpy as np
import torch


def calculate_rmsd(coords1, coords2, indices=None):
    """
    计算RMSD
    
    Args:
        coords1: 第一组坐标 (N, 3) - 可以是numpy或torch tensor
        coords2: 第二组坐标 (N, 3) - 可以是numpy或torch tensor
        indices: 可选，只计算指定位置的RMSD
        
    Returns:
        RMSD值（Å）
    """
    # 统一转换为numpy array（处理torch tensor）
    if torch.is_tensor(coords1):
        coords1 = coords1.detach().cpu().numpy()
    else:
        coords1 = np.array(coords1)
    
    if torch.is_tensor(coords2):
        coords2 = coords2.detach().cpu().numpy()
    else:
        coords2 = np.array(coords2)
    
    if indices is not None:
        coords1 = coords1[indices]
        coords2 = coords2[indices]
    
    # 确保形状匹配
    assert coords1.shape == coords2.shape, f"坐标形状不匹配: {coords1.shape} vs {coords2.shape}"
    
    # 计算RMSD
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd


def align_structures(coords1, coords2):
    """
    使用Kabsch算法对齐两个结构
    
    Args:
        coords1: 第一组坐标 (N, 3)
        coords2: 第二组坐标 (N, 3)
        
    Returns:
        aligned_coords1: 对齐后的第一组坐标
    """
    # 转换为numpy
    if torch.is_tensor(coords1):
        coords1 = coords1.detach().cpu().numpy()
    else:
        coords1 = np.array(coords1)
    
    if torch.is_tensor(coords2):
        coords2 = coords2.detach().cpu().numpy()
    else:
        coords2 = np.array(coords2)
    
    # 中心化
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    
    # 计算旋转矩阵
    H = coords1_centered.T @ coords2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 处理反射
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 应用旋转和平移
    aligned_coords1 = (R @ coords1_centered.T).T + centroid2
    
    return aligned_coords1


def sequence_identity(seq1, seq2):
    """
    计算两个序列的相同性
    
    Args:
        seq1: 第一个序列
        seq2: 第二个序列
        
    Returns:
        相同性（0-1之间）
    """
    if len(seq1) != len(seq2):
        # 如果长度不同，使用较短序列的长度
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1) if len(seq1) > 0 else 0.0


def extract_backbone_coords(protein, atom_name='CA'):
    """
    提取骨架原子坐标
    
    Args:
        protein: ESMProtein对象
        atom_name: 原子名称 ('CA', 'N', 'C')
        
    Returns:
        坐标数组 (N, 3)
    """
    if not hasattr(protein, 'coordinates') or protein.coordinates is None:
        return None
    
    coords = protein.coordinates
    
    # 如果是torch tensor，转换
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    else:
        coords = np.array(coords)
    
    # 假设坐标格式为 (L, num_atoms, 3)
    # CA原子通常是第二个（index=1）
    atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}
    
    if len(coords.shape) == 3:
        atom_idx = atom_indices.get(atom_name, 1)
        ca_coords = coords[:, atom_idx, :]
    else:
        # 已经是 (N, 3) 格式
        ca_coords = coords
    
    return ca_coords
