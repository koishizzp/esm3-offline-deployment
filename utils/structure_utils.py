"""
结构工具
用于处理蛋白质结构相关操作
"""

import numpy as np
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


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
    if torch is not None and torch.is_tensor(coords1):
        coords1 = coords1.detach().cpu().numpy()
    else:
        coords1 = np.array(coords1)
    
    if torch is not None and torch.is_tensor(coords2):
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
    """使用Kabsch算法对齐两个结构"""
    if torch is not None and torch.is_tensor(coords1):
        coords1 = coords1.detach().cpu().numpy()
    else:
        coords1 = np.array(coords1)
    
    if torch is not None and torch.is_tensor(coords2):
        coords2 = coords2.detach().cpu().numpy()
    else:
        coords2 = np.array(coords2)
    
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    
    H = coords1_centered.T @ coords2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    aligned_coords1 = (R @ coords1_centered.T).T + centroid2
    
    return aligned_coords1


def sequence_identity(seq1, seq2):
    """计算两个序列的相同性"""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1) if len(seq1) > 0 else 0.0


def extract_backbone_coords(protein, atom_name='CA'):
    """提取骨架原子坐标"""
    if not hasattr(protein, 'coordinates') or protein.coordinates is None:
        return None
    
    coords = protein.coordinates
    
    if torch is not None and torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    else:
        coords = np.array(coords)
    
    atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}
    
    if len(coords.shape) == 3:
        atom_idx = atom_indices.get(atom_name, 1)
        ca_coords = coords[:, atom_idx, :]
    else:
        ca_coords = coords
    
    return ca_coords


def load_pdb(pdb_file):
    """加载PDB文件"""
    try:
        import biotite.structure.io.pdb as pdb
        
        pdb_file = Path(pdb_file)
        pdb_data = pdb.PDBFile.read(str(pdb_file))
        structure = pdb.get_structure(pdb_data, model=1)
        
        chain_id = structure.chain_id[0]
        structure = structure[structure.chain_id == chain_id]
        
        from biotite.sequence import ProteinSequence
        from biotite.structure import get_residues

        residue_ids = get_residues(structure)[0]
        valid_res_ids = []
        sequence_letters = []

        for res_id in residue_ids:
            res_name = structure[structure.res_id == res_id].res_name[0]
            try:
                aa = ProteinSequence.convert_letter_3to1(res_name)
            except KeyError:
                # 跳过水分子/配体等非标准氨基酸残基（如 HOH）
                continue

            valid_res_ids.append(res_id)
            sequence_letters.append(aa)

        sequence = ''.join(sequence_letters)

        # 只保留有效残基中的CA坐标，确保序列与坐标长度一致
        ca_atoms = structure[(structure.atom_name == 'CA') & np.isin(structure.res_id, valid_res_ids)]
        coords = ca_atoms.coord
        
        return {
            'sequence': sequence,
            'coordinates': coords,
            'structure': structure
        }
    except Exception as e:
        print(f"加载PDB失败: {e}")
        raise


def save_to_fasta(sequence, output_file, header=">sequence", line_width=80):
    """保存序列到FASTA文件"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    header = str(header).strip()
    if not header.startswith('>'):
        header = f'>{header}'

    sequence = str(sequence).replace('\n', '').replace('\r', '').strip()

    with open(output_file, 'w') as f:
        f.write(f"{header}\n")
        for i in range(0, len(sequence), line_width):
            f.write(f"{sequence[i:i+line_width]}\n")



def load_from_fasta(fasta_file):
    """从FASTA文件加载序列
    
    Args:
        fasta_file: FASTA文件路径
        
    Returns:
        生成器，每次yield (header, sequence) 元组
    """
    fasta_file = Path(fasta_file)
    
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip().lstrip('>')
    sequence = ''.join(line.strip() for line in lines[1:])
    
    # 为了兼容性，返回生成器
    yield (header, sequence)
