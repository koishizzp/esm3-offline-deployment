#!/usr/bin/env python3
"""
交互式故障诊断向导
帮助用户一步步解决问题
"""

import os
import sys
import subprocess

def ask_yes_no(question):
    """询问是/否问题"""
    while True:
        response = input(f"{question} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("请输入 y 或 n")

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def run_command(cmd, description=None):
    """运行命令并显示输出"""
    if description:
        print(f"\n{description}")
    
    print(f"运行: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, 
            capture_output=True, text=True, 
            timeout=30
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"命令执行失败: {e}")
        return False

def diagnose_environment():
    """诊断环境问题"""
    print_header("环境诊断")
    
    # 检查conda
    print("\n[1/4] 检查Conda环境...")
    if run_command("conda env list | grep esm3_env"):
        print("✓ esm3_env环境存在")
        
        if ask_yes_no("是否已激活esm3_env环境？"):
            print("✓ 环境已激活")
        else:
            print("\n请运行以下命令激活环境:")
            print("  conda activate esm3_env")
            print("\n然后重新运行此脚本")
            return False
    else:
        print("✗ esm3_env环境不存在")
        print("\n请先完成ESM3部署:")
        print("  参考之前的部署文档")
        return False
    
    # 检查Python
    print("\n[2/4] 检查Python...")
    run_command("python --version")
    run_command("which python")
    
    # 检查ESM
    print("\n[3/4] 检查ESM模块...")
    success = run_command(
        'python -c "import esm; print(esm.__file__)"',
        "尝试导入ESM..."
    )
    
    if not success:
        print("\n✗ ESM模块导入失败")
        if ask_yes_no("是否尝试添加ESM到Python路径？"):
            run_command("export PYTHONPATH=/mnt/disk3/tio_nekton4/esm3/esm:$PYTHONPATH")
            print("请将以下命令添加到 ~/.bashrc:")
            print("  export PYTHONPATH=/mnt/disk3/tio_nekton4/esm3/esm:$PYTHONPATH")
    else:
        print("✓ ESM模块正常")
    
    # 检查CUDA
    print("\n[4/4] 检查CUDA...")
    cuda_ok = run_command(
        'python -c "import torch; print(f\'CUDA: {torch.cuda.is_available()}\')"'
    )
    
    if not cuda_ok:
        print("\n⚠ CUDA可能不可用")
        if ask_yes_no("是否检查GPU状态？"):
            run_command("nvidia-smi")
    
    return True

def diagnose_files():
    """诊断文件问题"""
    print_header("文件诊断")
    
    # 检查项目结构
    print("\n[1/3] 检查项目结构...")
    dirs = ['data/templates', 'data/prompts', 'data/candidates', 'data/results']
    
    for d in dirs:
        if os.path.exists(d):
            print(f"  ✓ {d}")
        else:
            print(f"  ✗ {d} 不存在")
            if ask_yes_no(f"是否创建 {d}？"):
                os.makedirs(d, exist_ok=True)
                print(f"    → 已创建 {d}")
    
    # 检查模型文件
    print("\n[2/3] 检查模型文件...")
    model_dir = "/mnt/disk3/tio_nekton4/esm3/weights"
    
    if os.path.exists(model_dir):
        print(f"  ✓ 模型目录存在")
        run_command(f"ls -lh {model_dir}/*.pth | head -5")
    else:
        print(f"  ✗ 模型目录不存在: {model_dir}")
        print("\n请检查ESM3是否正确部署")
        return False
    
    # 检查模板
    print("\n[3/3] 检查PDB模板...")
    template = "data/templates/1QY3.pdb"
    
    if os.path.exists(template):
        size = os.path.getsize(template)
        print(f"  ✓ 模板存在 ({size} bytes)")
        
        if size < 1000:
            print("  ⚠ 文件太小，可能损坏")
            if ask_yes_no("是否重新下载？"):
                run_command("python scripts/01_download_template.py")
    else:
        print(f"  ✗ 模板不存在")
        if ask_yes_no("是否现在下载？"):
            run_command("python scripts/01_download_template.py")
    
    return True

def diagnose_runtime():
    """诊断运行时问题"""
    print_header("运行时诊断")
    
    print("\n请选择遇到的问题类型:")
    print("1. 生成速度很慢")
    print("2. 内存/显存不足")
    print("3. 生成的序列质量差")
    print("4. 评估指标全是None")
    print("5. 其他问题")
    
    choice = input("\n请输入数字 (1-5): ").strip()
    
    if choice == "1":
        print("\n=== 生成速度慢 ===")
        print("可能原因:")
        print("1. 没有使用GPU")
        print("2. 步数设置太高")
        print("3. 第一次运行（模型加载慢）")
        
        if ask_yes_no("是否检查GPU使用？"):
            run_command("nvidia-smi")
        
        print("\n建议:")
        print("- 降低生成步数（在config.py中）")
        print("- 确保CUDA可用")
        print("- 首次运行耐心等待3-5分钟")
    
    elif choice == "2":
        print("\n=== 内存/显存不足 ===")
        run_command("free -h", "检查内存:")
        run_command("nvidia-smi", "检查显存:")
        
        print("\n建议:")
        print("- 降低batch_size: --batch-size 1")
        print("- 降低生成步数")
        print("- 关闭其他占用GPU的程序")
        
        if ask_yes_no("是否现在清理GPU缓存？"):
            run_command('python -c "import torch; torch.cuda.empty_cache()"')
    
    elif choice == "3":
        print("\n=== 序列质量差 ===")
        print("检查生成的序列...")
        
        if os.path.exists("data/candidates/candidate_0.fasta"):
            run_command("head -5 data/candidates/candidate_0.fasta")
            
            print("\n建议:")
            print("- 检查prompt是否正确")
            print("- 调整温度参数（0.5-1.0）")
            print("- 增加生成步数")
        else:
            print("未找到候选文件，请先生成")
    
    elif choice == "4":
        print("\n=== 评估指标None ===")
        print("这通常是因为结构预测方法问题")
        
        print("\n建议:")
        print("- 使用简化评估（只评估序列）")
        print("- 修改05_evaluate_candidates.py")
        print("- 参考TROUBLESHOOTING.md场景4")
    
    else:
        print("\n=== 通用故障排除 ===")
        print("建议步骤:")
        print("1. 运行完整诊断: python scripts/diagnostic.py")
        print("2. 查看DEBUG_GUIDE.md")
        print("3. 查看TROUBLESHOOTING.md")
        print("4. 保存错误日志求助")

def main():
    """主函数"""
    print_header("GFP复现项目 - 交互式故障诊断")
    print("\n此工具将帮助您诊断和解决常见问题")
    print("请按照提示回答问题")
    
    # 选择诊断类型
    print("\n请选择诊断类型:")
    print("1. 环境问题（无法运行脚本）")
    print("2. 文件问题（找不到文件）")
    print("3. 运行时问题（运行中出错）")
    print("4. 完整诊断（运行所有检查）")
    
    choice = input("\n请输入数字 (1-4): ").strip()
    
    if choice == "1":
        if diagnose_environment():
            print("\n✓ 环境诊断完成")
        else:
            print("\n请按建议修复环境问题")
    
    elif choice == "2":
        if diagnose_files():
            print("\n✓ 文件诊断完成")
        else:
            print("\n请按建议修复文件问题")
    
    elif choice == "3":
        diagnose_runtime()
    
    elif choice == "4":
        print("\n运行完整诊断...")
        diagnose_environment()
        diagnose_files()
        diagnose_runtime()
    
    else:
        print("无效选择")
        return
    
    # 结束
    print_header("诊断完成")
    print("\n有用的资源:")
    print("- DEBUG_GUIDE.md - 详细调试指南")
    print("- TROUBLESHOOTING.md - 真实场景解决方案")
    print("- scripts/diagnostic.py - 自动诊断脚本")
    print("\n如需进一步帮助，请保存以下信息:")
    print("  python scripts/diagnostic.py > diagnostic_report.txt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消诊断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
