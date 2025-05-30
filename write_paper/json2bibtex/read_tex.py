import re


def extract_arxiv_ids_from_file(paper_folder, file_name):

    with open(paper_folder + '/' + file_name, 'r', encoding='utf-8') as file:  # 打开 tex 文件
        text = file.read()  # 读取全部文本内容

    # 匹配 \cite{} 中的内容的正则表达式
    pattern = r'\\cite\{([^}]*)\}'
    matches = re.findall(pattern, text)  # 匹配所有 \cite{} 中的内容

    # 提取每个 \cite{} 中的内容，可能包含多个引用，用逗号或空格分隔
    arxiv_ids = set()
    for match in matches:
        # 分割多个引用，去除空白
        refs = re.split(r'[ ,]+', match)
        for ref in refs:
            if ref:
                arxiv_ids.add(ref)

    return arxiv_ids  # 返回编号集合


if __name__ == '__main__':

    pass
