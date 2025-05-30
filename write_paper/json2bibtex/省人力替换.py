import os
import re


def modify_latex_file(file_name):
    """
    读取LaTeX文件并进行指定替换
    
    Args:
        file_name (str): 要修改的LaTeX文件路径
    """
    print(f"正在处理文件: {file_name}")

    def convert_to_title_case(title):
        """
        将字符串转换为标题格式:
        1. 将 '&' 替换为 'and'
        2. 除了特定的连接词外，所有单词的首字母大写
        3. 连字符后的单词首字母大写
        4. 保留原始字符串中特定词的大写形式（如缩写词）
        """
        # 检查空字符串
        if not title:
            return ""
            
        # 替换 '&' 为 'and'
        title = title.replace('&', 'and')
        
        # 定义应保持小写的词（除非是第一个词或连字符后的词）
        lowercase_words = {'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for',
                           'in', 'to', 'of', 'at', 'by', 'on'}
        
        words = title.split()
        result = []
        
        for i, word in enumerate(words):
            # 如果单词包含连字符
            if '-' in word:
                hyphenated_parts = word.split('-')
                processed_parts = []
                
                for j, part in enumerate(hyphenated_parts):
                    # 检查是否应该保留原始大小写
                    if should_preserve_case(part):
                        processed_parts.append(part)
                    # 连字符后的部分总是大写首字母
                    elif j > 0:
                        processed_parts.append(part.capitalize())
                    # 第一部分依照一般规则处理
                    elif i == 0 or part.lower() not in lowercase_words:
                        processed_parts.append(part.capitalize())
                    else:
                        processed_parts.append(part.lower())
                
                result.append('-'.join(processed_parts))
            else:
                # 检查是否应该保留原始大小写
                if should_preserve_case(word):
                    result.append(word)
                # 正常单词处理
                elif i == 0 or word.lower() not in lowercase_words:
                    result.append(word.capitalize())
                else:
                    result.append(word.lower())
        
        return ' '.join(result)


    def should_preserve_case(word):
        """
        检查是否应该保留单词的原始大小写格式。
        判断依据：
        1. 全大写的单词（可能是缩写词）
        2. 内部有大写字母的单词（如 'iPhone'）
        """
        # 忽略空字符串和单个字符
        if not word or len(word) <= 1:
            return False
        
        # 如果单词全是大写，可能是缩写词
        if word.isupper():
            return True
        
        # 检查单词中除首字母外是否有其他大写字母
        if any(c.isupper() for c in word[1:]):
            return True
        
        return False

    
    try:
        # 读取文件
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 处理文件名以创建标题
        base_name = os.path.splitext(os.path.basename(file_name))[0]  # 获取不带扩展名的文件名
        title = base_name.replace('_', ' ')  # 替换下划线为空格
        # 只在这里使用正则表达式来处理多个空格
        title = re.sub(r'\s+', ' ', title)   # 将多个空格替换为单个空格
        title = title.replace('survey', '')
        title = convert_to_title_case(title)
        title = title + ": A Comprehensive Survey"
        
        # 检查是否已经完成替换
        if title in content and '\\usepackage{tikz}' in content:
            print("替换已完成，无需再次执行。")
            return

        # 替换0: 去掉 cite 包
        content = content.replace('\\usepackage{cite}\n', '')
        
        # 替换1: 不使用正则表达式
        old_text1 = '% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.'
        new_text1 = '''
\\usepackage{natbib}
\\usepackage{tikz}
\\usepackage{float}
\\usepackage{forest}
\\usetikzlibrary{arrows.meta, shapes.geometric, positioning, fit, backgrounds}
\\usepackage{booktabs}

\\usepackage{hyperref}  % Add hyperref for styling links
\\hypersetup{
    colorlinks=true,
    linkcolor=darkblue,  % Customize the color of the internal links (like citations)
    citecolor=darkblue,  % Customize the color of citations
    urlcolor=darkblue  % Customize the color of URLs
}
\\definecolor{darkblue}{rgb}{0.0, 0.0, 0.55}  % Define dark blue color
'''
        content = content.replace(old_text1, new_text1)
        
        # 替换2: 不使用正则表达式
        old_text2 = '\\title{\n\\thanks{Identify applicable funding agency here. If none, delete this.}\n}'
        new_text2 = f'''\\title{{{title}
%
\\thanks{{\\hspace*{{-\\parindent}}\\rule{{3.8cm}}{{0.4pt}} \\\\ 
$\\ast$: Equal Contribution;  $\\dagger$: Corresponding Author.}}
}}'''
        
        # 尝试各种可能的格式
        if old_text2 not in content:
            old_text2 = '\\title{\\thanks{Identify applicable funding agency here. If none, delete this.}}'
            if old_text2 not in content:
                # 尝试不带换行和空格的版本
                old_text2 = '\\title{\\thanks{Identify applicable funding agency here. If none, delete this.}}'
        
        content = content.replace(old_text2, new_text2)
        
        # 添加参考文献设置
        if '\\bibliographystyle{plain}' not in content:
            content = content.replace(
                '\\end{document}',
                f'''\\bibliographystyle{{apalike}}
\\bibliography{base_name}

\\end{{document}}'''
            )
        
        # 新要求1: 确保 \title{ 后面没有空格
        content = content.replace('\\title{ ', '\\title{')
        
        # 新要求2和3: 在所有包含 \section{ 的行前添加两个换行，后添加一个换行
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            if '\\section{' in lines[i]:
                # 在当前位置插入两个空行
                lines.insert(i, '')
                lines.insert(i, '')
                # 由于插入了两行，当前行索引变为 i+2
                i += 2
                # 在当前行末尾添加换行
                lines[i] = lines[i] + '\n'
            i += 1
        content = '\n'.join(lines)
        
        # 将修改后的内容写回文件
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(content)
        
        # print(f"文件修改完成")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_name}'")
    except Exception as e:
        print(f"发生错误：{str(e)}")


def list_tex_files(folder_path):
    """
    列出指定文件夹中的所有.tex文件
    
    Args:
        folder_path (str): 要搜索.tex文件的文件夹路径
        
    Returns:
        list: .tex文件路径列表
    """
    tex_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))
    return tex_files


def process_all_tex_files(folder_path):
    """
    处理指定文件夹中的所有.tex文件
    
    Args:
        folder_path (str): 包含.tex文件的文件夹路径
    """
    tex_files = list_tex_files(folder_path)
    
    if not tex_files:
        print(f"在 {folder_path} 文件夹中未找到任何 .tex 文件")
        return
    
    print(f"在 {folder_path} 文件夹中找到 {len(tex_files)} 个 .tex 文件")
    
    for tex_file in tex_files:
        modify_latex_file(tex_file)


if __name__ == '__main__':
    
    folder_path = "paper"
    print("-" * 50)
    print(f"开始处理文件夹: {folder_path}")
    print("-" * 50)
    process_all_tex_files(folder_path)
    print("-" * 50)

    input('已完成')
