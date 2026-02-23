#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESM3嵌入提取 - Web界面版 (兼容Gradio 6.0)
"""

import gradio as gr
import subprocess
import os
from pathlib import Path
import pandas as pd

# ============================================================
# 配置
# ============================================================

SCRIPT_PATH = "get_embeddings_offline.py"
DEFAULT_OUTPUT_DIR = "embeddings_output"

# ============================================================
# 核心函数
# ============================================================

def extract_embeddings(
    input_file,
    output_dir,
    use_half_precision,
    max_length,
    output_format,
    device,
    enable_resume
):
    """执行嵌入提取"""
    
    if input_file is None:
        return "❌ 错误：请先上传FASTA文件", None, None
    
    input_path = input_file.name
    
    cmd = ["python", SCRIPT_PATH, input_path, "-o", output_dir]
    
    if use_half_precision:
        cmd.append("--half")
    
    if max_length:
        cmd.extend(["--max-length", str(max_length)])
    
    if output_format:
        cmd.extend(["--format", output_format])
    
    if device:
        cmd.extend(["--device", device])
    
    if not enable_resume:
        cmd.append("--no-resume")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        output_text = result.stdout + "\n" + result.stderr
        
        if result.returncode == 0:
            summary = analyze_results(output_dir)
            status = f"✅ 完成！\n\n{summary}"
            
            output_files = list(Path(output_dir).glob("*.pkl.gz"))
            if len(output_files) <= 10:
                file_list = "\n".join([f.name for f in output_files[:10]])
                return status, output_text, file_list
            else:
                return status, output_text, f"生成了 {len(output_files)} 个文件"
        else:
            return f"❌ 失败：\n{output_text}", output_text, None
            
    except subprocess.TimeoutExpired:
        return "⏰ 超时：任务运行超过1小时", None, None
    except Exception as e:
        return f"❌ 错误：{str(e)}", None, None


def analyze_results(output_dir):
    """分析输出结果"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return "输出目录不存在"
    
    pkl_files = list(output_path.glob("*.pkl.gz"))
    npy_files = list(output_path.glob("*.npy"))
    
    summary = f"成功生成: {len(pkl_files) + len(npy_files)} 个嵌入文件\n"
    summary += f"  - PKL格式: {len(pkl_files)} 个\n"
    summary += f"  - NPY格式: {len(npy_files)} 个\n"
    
    failed_csv = output_path / "failed_sequences.csv"
    if failed_csv.exists():
        try:
            df = pd.read_csv(failed_csv)
            failed_count = len(df)
            if failed_count > 0:
                summary += f"\n⚠️ 失败: {failed_count} 个序列\n"
        except:
            pass
    
    return summary


def get_file_info(input_file):
    """获取上传文件信息"""
    if input_file is None:
        return "未上传文件"
    
    try:
        with open(input_file.name, 'r') as f:
            seq_count = sum(1 for line in f if line.startswith('>'))
        
        file_size = os.path.getsize(input_file.name) / 1024 / 1024
        
        return f"""📄 文件信息:
  文件名: {os.path.basename(input_file.name)}
  大小: {file_size:.2f} MB
  序列数: {seq_count}
"""
    except Exception as e:
        return f"无法读取文件信息: {e}"


# ============================================================
# Gradio界面
# ============================================================

def create_interface():
    """创建Web界面"""
    
    with gr.Blocks(title="ESM3嵌入提取工具") as demo:
        
        gr.Markdown("""
        # 🧬 ESM3蛋白质序列嵌入提取工具
        
        通过Web界面轻松提取蛋白质序列的ESM3嵌入向量
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📥 输入文件")
                
                input_file = gr.File(
                    label="上传FASTA文件",
                    file_types=[".faa", ".fasta", ".fa"],
                    type="filepath"
                )
                
                file_info = gr.Textbox(
                    label="文件信息",
                    lines=4,
                    interactive=False
                )
                
                input_file.change(
                    fn=get_file_info,
                    inputs=[input_file],
                    outputs=[file_info]
                )
                
                gr.Markdown("## ⚙️ 参数设置")
                
                output_dir = gr.Textbox(
                    label="输出目录",
                    value=DEFAULT_OUTPUT_DIR
                )
                
                use_half = gr.Checkbox(
                    label="启用半精度加速 (推荐！)",
                    value=True
                )
                
                max_length = gr.Slider(
                    label="最大序列长度",
                    minimum=128,
                    maximum=4096,
                    value=4096,
                    step=128,
                    info="超过此长度的序列将被跳过"
                )
                
                output_format = gr.Radio(
                    label="输出格式",
                    choices=["pkl.gz", "npy", "both"],
                    value="pkl.gz"
                )
                
                device = gr.Radio(
                    label="计算设备",
                    choices=["auto", "cuda", "cpu"],
                    value="auto"
                )
                
                enable_resume = gr.Checkbox(
                    label="启用断点续传",
                    value=True
                )
                
                submit_btn = gr.Button(
                    "🚀 开始提取",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 📊 运行状态")
                
                status_box = gr.Textbox(
                    label="状态",
                    lines=5,
                    interactive=False
                )
                
                gr.Markdown("## 📝 详细日志")
                
                log_box = gr.Textbox(
                    label="运行日志",
                    lines=15,
                    interactive=False
                )
                
                gr.Markdown("## 📁 输出文件")
                
                files_box = gr.Textbox(
                    label="生成的文件",
                    lines=8,
                    interactive=False
                )
        
        submit_btn.click(
            fn=extract_embeddings,
            inputs=[
                input_file,
                output_dir,
                use_half,
                max_length,
                output_format,
                device,
                enable_resume
            ],
            outputs=[status_box, log_box, files_box]
        )
        
        gr.Markdown("""
        ---
        ### 💡 使用提示
        
        1. 上传FASTA文件
        2. 调整参数（推荐启用半精度）
        3. 点击"开始提取"
        4. 等待处理完成
        
        ### ⚡ 预计时间
        - <1000条: 1-5分钟
        - 1000-10000条: 5-30分钟
        """)
    
    return demo


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    try:
        import gradio
        print("✓ Gradio已安装")
    except ImportError:
        print("❌ 请先安装: pip install gradio")
        exit(1)
    
    if not os.path.exists(SCRIPT_PATH):
        print(f"⚠️  找不到 {SCRIPT_PATH}")
    
    demo = create_interface()
    
    print("\n" + "="*60)
    print("🚀 ESM3嵌入提取Web界面")
    print("="*60)
    print("\n访问地址将在下方显示...")
    print("按 Ctrl+C 停止\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
