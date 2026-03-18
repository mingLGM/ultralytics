# import re
# import markdown
# from docx import Document
# from docx.shared import Pt, Inches
# from docx.enum.text import WD_ALIGN_PARAGRAPH
# from docx.oxml.ns import qn
# from latex2mathml.converter import convert as latex2mathml

# # 配置中文字体支持
# def set_font(run, font_name='宋体', size=12):
#     run.font.name = font_name
#     run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)
#     run.font.size = Pt(size)

# def parse_markdown_to_docx(md_content, output_path):
#     doc = Document()
    
#     # 设置全局样式
#     style = doc.styles['Normal']
#     font = style.font
#     font.name = '宋体'
#     font._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
#     font.size = Pt(12)

#     lines = md_content.split('\n')
#     current_paragraph = None
    
#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()
        
#         # 处理标题
#         if line.startswith('# '):
#             p = doc.add_heading(line[2:], level=1)
#             set_font(p.runs[0], '黑体', 16)
#         elif line.startswith('## '):
#             p = doc.add_heading(line[3:], level=2)
#             set_font(p.runs[0], '黑体', 14)
#         elif line.startswith('### '):
#             p = doc.add_heading(line[4:], level=3)
#             set_font(p.runs[0], '黑体', 13)
#         elif line.startswith('#### '):
#             p = doc.add_heading(line[5:], level=4)
#             set_font(p.runs[0], '宋体', 12)
        
#         # 处理公式块 ($$ ... $$)
#         elif line.startswith('$$'):
#             equation_content = []
#             # 收集多行公式直到遇到结束的 $$
#             if not line.endswith('$$') or line == '$$':
#                 # 如果是单行 $$ 开头，可能需要向后查找
#                 temp_line = line[2:]
#                 if temp_line.strip(): equation_content.append(temp_line)
#                 i += 1
#                 while i < len(lines):
#                     if '$$' in lines[i]:
#                         end_part = lines[i].split('$$')[0]
#                         if end_part.strip(): equation_content.append(end_part)
#                         break
#                     equation_content.append(lines[i])
#                     i += 1
#             else:
#                 # 单行公式 $$...$$
#                 equation_content.append(line[2:-2])
            
#             latex_str = '\n'.join(equation_content).strip()
#             if latex_str:
#                 # 尝试转换 LaTeX 为 MathML (Word 支持 OMML/MathML)
#                 # 注意：python-docx 对原生 MathML 支持有限，这里我们将其作为文本插入，
#                 # 或者在实际高级应用中调用 Word 的 Equation API。
#                 # 为了兼容性，这里我们将 LaTeX 源码保留在文档中，并用特殊格式标记，
#                 # 或者尝试简单的转换。
#                 # *最佳实践*：在 Word 中，我们可以插入一个包含 LaTeX 的段落，
#                 # 用户可以使用 Word 的 "Alt + =" 功能自动识别，或者我们尝试转换。
                
#                 p = doc.add_paragraph()
#                 p.alignment = WD_ALIGN_PARAGRAPH.CENTER
#                 run = p.add_run(f"[公式]: {latex_str}")
#                 set_font(run, 'Times New Roman', 12)
#                 run.italic = True
#                 # 注：完全自动将复杂 LaTeX 转为 Word 原生公式对象需要复杂的 XML 操作，
#                 # 此处为了稳定性，将公式以清晰的 LaTeX 文本形式保留，便于后续一键转换或在支持 LaTeX 的编辑器中使用。
        
#         # 处理普通文本和列表
#         elif line:
#             # 处理列表
#             if line.startswith('- ') or line.startswith('* '):
#                 p = doc.add_paragraph(style='List Bullet')
#                 content = line[2:]
#             elif re.match(r'^\d+\.\s', line):
#                 p = doc.add_paragraph(style='List Number')
#                 content = re.sub(r'^\d+\.\s', '', line)
#             else:
#                 p = doc.add_paragraph()
#                 content = line
            
#             # 处理行内公式 ($ ... $)
#             # 简单分割处理
#             parts = re.split(r'(\$[^$]+\$)', content)
#             for part in parts:
#                 if part.startswith('$') and part.endswith('$'):
#                     run = p.add_run(part)
#                     set_font(run, 'Times New Roman', 12)
#                     run.font.italic = True
#                 else:
#                     run = p.add_run(part)
#                     set_font(run, '宋体', 12)
        
#         i += 1

#     doc.save(output_path)
#     print(f"成功转换文档至: {output_path}")

# # 读取 markdown 文件
# with open(r'E:\work\code\ultralytics\comutils\专利交底书.md', 'r', encoding='utf-8') as f:
#     md_text = f.read()

# # 执行转换
# parse_markdown_to_docx(md_text, '专利交底书_转换版.docx')












import pypandoc
import os

def convert_md_to_docx_with_equations(input_file, output_file):
    """
    使用 Pandoc 将 Markdown (含 LaTeX 公式) 转换为 Word (含原生公式)
    """
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    print(f"正在转换: {input_file} -> {output_file} ...")
    print("注意: 此过程需要系统已安装 'pandoc' 工具。")

    try:
        # extra_args 配置说明:
        # --standalone: 生成完整的文档结构
        # --mathml: 某些版本需要，但通常 pandoc 默认对 docx 处理很好
        # --reference-doc: (可选) 如果你想套用特定的 Word 模板，可以加上这个参数
        output = pypandoc.convert_file(
            input_file, 
            'docx', 
            extra_args=[
                '--standalone',
                '--mathjax', # 确保数学公式被正确解析
            ],
            outputfile=output_file
        )
        
        # 如果 convert_file 返回空字符串且没有报错，通常表示成功写入文件
        # pypandoc 在指定 outputfile 时直接写入磁盘
        print(f"✅ 转换成功！")
        print(f"📄 文件位置: {os.path.abspath(output_file)}")
        print("\n💡 提示: 打开 Word 文档，点击任意公式，应该能看到‘公式工具’选项卡，说明已是原生格式。")

    except OSError as e:
        print("❌ 转换失败：未找到 Pandoc 程序。")
        print("请前往 https://pandoc.org/installing.html 下载并安装 Pandoc，然后重试。")
        print(f"详细错误: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

# 配置文件名
input_md = r'E:\work\code\ultralytics\comutils\专利交底书.md'
output_docx = r'E:\work\code\ultralytics\comutils\专利交底书_原生公式版.docx'

# 执行转换
convert_md_to_docx_with_equations(input_md, output_docx)