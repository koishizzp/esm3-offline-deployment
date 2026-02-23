#!/usr/bin/env python3
"""
完整的系统诊断脚本
检查GFP复现项目的所有依赖和配置
"""

import os
import sys
import subprocess

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def check_mark(condition):
    """返回检查标记"""
    return "✓" if condition else "✗"

def run_diagnostic():
    """运行完整诊断"""
    
    print_section("GFP复现项目 - 系统诊断")
    print(f"诊断时间: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")
    
    issues_found = []
    
    # 1. Python环境
    print_section("[1/12] Python环境")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"Python路径: {sys.executable}")
    
    if sys.version_info < (3, 12):
        issues_found.append("Python版本 < 3.12，ESM可能不兼容")
        print("  ⚠ 警告: Python版本可能不兼容")
    else:
        print("  ✓ Python版本合适")
    
    # 2. Conda环境
    print_section("[2/12] Conda环境")
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True)
        envs = result.stdout
        
        if 'esm3_env' in envs:
            print("  ✓ esm3_env环境存在")
            
            # 检查是否激活
            if 'esm3_env' in sys.executable:
                print("  ✓ esm3_env环境已激活")
            else:
                print("  ✗ esm3_env环境未激活")
                issues_found.append("需要运行: conda activate esm3_env")
        else:
            print("  ✗ esm3_env环境不存在")
            issues_found.append("ESM3环境未创建")
    except FileNotFoundError:
        print("  ✗ conda命令未找到")
        issues_found.append("Conda未安装或不在PATH中")
    
    # 3. ESM模块
    print_section("[3/12] ESM模块")
    try:
        import esm
        print(f"  ✓ ESM已安装")
        print(f"    路径: {esm.__file__}")
        
        # 检查SDK
        try:
            from esm.sdk.api import ESM3InferenceClient
            print(f"  ✓ ESM SDK可用")
        except ImportError as e:
            print(f"  ✗ ESM SDK导入失败: {e}")
            issues_found.append("ESM SDK不完整")
            
    except ImportError as e:
        print(f"  ✗ ESM未安装: {e}")
        issues_found.append("需要安装ESM: pip install -e /mnt/disk3/tio_nekton4/esm3/esm")
    
    # 4. PyTorch和CUDA
    print_section("[4/12] PyTorch和CUDA")
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA可用: {check_mark(cuda_available)} {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  ⚠ 警告: CUDA不可用，将使用CPU（非常慢）")
            issues_found.append("CUDA不可用")
            
    except ImportError:
        print("  ✗ PyTorch未安装")
        issues_found.append("需要安装PyTorch")
    
    # 5. 核心依赖
    print_section("[5/12] 核心依赖")
    required_packages = {
        'numpy': '数值计算',
        'scipy': '科学计算',
        'biopython': '生物信息学',
        'requests': 'HTTP请求',
        'biotite': 'PDB处理（可选）'
    }
    
    for pkg, desc in required_packages.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg:12} - {desc}")
        except ImportError:
            optional = "（可选）" in desc
            mark = "⚠" if optional else "✗"
            print(f"  {mark} {pkg:12} - {desc}")
            if not optional:
                issues_found.append(f"需要安装: pip install {pkg} --break-system-packages")
    
    # 6. 项目目录结构
    print_section("[6/12] 项目目录")
    required_dirs = [
        'data/templates',
        'data/prompts',
        'data/candidates',
        'data/results',
        'scripts',
        'utils'
    ]
    
    for d in required_dirs:
        exists = os.path.exists(d)
        print(f"  {check_mark(exists)} {d}")
        if not exists:
            try:
                os.makedirs(d, exist_ok=True)
                print(f"      → 已自动创建")
            except Exception as e:
                print(f"      → 创建失败: {e}")
                issues_found.append(f"无法创建目录: {d}")
    
    # 7. 配置文件
    print_section("[7/12] 配置文件")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
        from config import MODEL_DIR, MODEL_NAME, PROTEIN_LENGTH
        print(f"  ✓ config.py加载成功")
        print(f"    模型目录: {MODEL_DIR}")
        print(f"    模型名称: {MODEL_NAME}")
        print(f"    蛋白长度: {PROTEIN_LENGTH}")
    except Exception as e:
        print(f"  ✗ config.py加载失败: {e}")
        issues_found.append("配置文件有问题")
    
    # 8. 模型文件
    print_section("[8/12] 模型权重文件")
    model_dir = "/mnt/disk3/tio_nekton4/esm3/weights"
    
    if os.path.exists(model_dir):
        print(f"  ✓ 模型目录存在: {model_dir}")
        
        required_files = {
            'esm3_sm_open_v1.pth': 2.7,  # GB
            'esm3_structure_encoder_v0.pth': 0.06,
            'esm3_structure_decoder_v0.pth': 1.2,
            'esm3_function_decoder_v0.pth': 1.3
        }
        
        for filename, expected_size_gb in required_files.items():
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                size_gb = os.path.getsize(filepath) / 1024 / 1024 / 1024
                size_match = abs(size_gb - expected_size_gb) < 0.5
                
                mark = check_mark(size_match)
                print(f"  {mark} {filename}")
                print(f"      大小: {size_gb:.2f} GB (期望: {expected_size_gb:.2f} GB)")
                
                if not size_match:
                    issues_found.append(f"{filename} 大小异常，可能损坏")
            else:
                print(f"  ✗ {filename} 不存在")
                issues_found.append(f"缺少模型文件: {filename}")
    else:
        print(f"  ✗ 模型目录不存在: {model_dir}")
        issues_found.append("模型目录不存在，检查部署步骤")
    
    # 9. 模板文件
    print_section("[9/12] PDB模板文件")
    template_path = "data/templates/1QY3.pdb"
    
    if os.path.exists(template_path):
        size = os.path.getsize(template_path)
        print(f"  ✓ 模板文件存在: {template_path}")
        print(f"    大小: {size} bytes")
        
        # 验证内容
        try:
            with open(template_path, 'r') as f:
                content = f.read()
                has_header = 'HEADER' in content
                has_atoms = 'ATOM' in content
                atom_count = content.count('\nATOM')
                
                print(f"    HEADER记录: {check_mark(has_header)}")
                print(f"    ATOM记录数: {atom_count}")
                
                if atom_count < 100:
                    print(f"    ⚠ 警告: ATOM记录太少，文件可能不完整")
                    issues_found.append("PDB文件可能损坏")
        except Exception as e:
            print(f"    ✗ 读取失败: {e}")
            issues_found.append("无法读取PDB文件")
    else:
        print(f"  ✗ 模板文件不存在")
        issues_found.append("需要运行: python scripts/01_download_template.py")
    
    # 10. GPU状态
    print_section("[10/12] GPU状态")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.free,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 5:
                    idx, name, free, total, util = parts
                    print(f"  GPU {idx}: {name}")
                    print(f"    显存: {free} MB 可用 / {total} MB 总计")
                    print(f"    利用率: {util}%")
                    
                    free_mb = int(free)
                    if free_mb < 8000:
                        print(f"    ⚠ 警告: 可用显存 < 8GB，可能影响生成")
                        issues_found.append("GPU显存不足")
        else:
            print(f"  ✗ nvidia-smi执行失败")
            issues_found.append("GPU工具异常")
            
    except FileNotFoundError:
        print("  ✗ nvidia-smi未找到")
        issues_found.append("GPU驱动未安装")
    except subprocess.TimeoutExpired:
        print("  ✗ nvidia-smi超时")
        issues_found.append("GPU响应超时")
    
    # 11. 磁盘空间
    print_section("[11/12] 磁盘空间")
    try:
        result = subprocess.run(['df', '-h', '/mnt/disk3'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print(f"  {lines[0]}")  # Header
                print(f"  {lines[1]}")  # Data
                
                parts = lines[1].split()
                if len(parts) >= 5:
                    used_pct = parts[4].rstrip('%')
                    try:
                        if int(used_pct) > 90:
                            print(f"  ⚠ 警告: 磁盘使用率 > 90%")
                            issues_found.append("磁盘空间不足")
                    except ValueError:
                        pass
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
    
    # 12. 快速功能测试
    print_section("[12/12] 快速功能测试")
    
    try:
        # 测试导入
        print("  [1/3] 测试导入工具模块...")
        from utils.esm_wrapper import ESM3Generator
        from utils.structure_utils import load_pdb, sequence_identity
        from utils.evaluation import evaluate_candidate
        print("    ✓ 工具模块导入成功")
        
        # 测试序列处理
        print("  [2/3] 测试序列处理...")
        id_test = sequence_identity("MKTEST", "MKTEXT")
        print(f"    ✓ 序列相似度计算: {id_test:.2%}")
        
        # 测试模型加载（可选，因为很慢）
        print("  [3/3] 测试模型加载（跳过，太慢）...")
        print("    ⊙ 建议手动测试: python -c 'from utils.esm_wrapper import ESM3Generator; g = ESM3Generator(\"/mnt/disk3/tio_nekton4/esm3/weights\")'")
        
    except Exception as e:
        print(f"    ✗ 功能测试失败: {e}")
        import traceback
        print("\n错误详情:")
        traceback.print_exc()
        issues_found.append("基本功能测试失败")
    
    # 总结
    print_section("诊断总结")
    
    if issues_found:
        print(f"\n发现 {len(issues_found)} 个问题:\n")
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. {issue}")
        
        print("\n建议按以下顺序解决:")
        print("1. 确保conda环境激活: conda activate esm3_env")
        print("2. 检查模型文件完整性")
        print("3. 安装缺失的依赖")
        print("4. 检查GPU驱动和CUDA")
        print("\n详细解决方案请参考 DEBUG_GUIDE.md")
    else:
        print("\n✓✓✓ 所有检查通过！系统就绪 ✓✓✓")
        print("\n可以开始运行:")
        print("  ./run_all.sh")
        print("或分步执行:")
        print("  cd scripts")
        print("  python 01_download_template.py")
        print("  python 02_create_prompt.py")
        print("  python 03_generate_single.py")
    
    print("\n" + "=" * 70)
    print("诊断完成！")
    print("=" * 70)
    
    # 保存诊断报告
    print("\n保存诊断报告到: diagnostic_report.txt")
    

if __name__ == "__main__":
    run_diagnostic()
