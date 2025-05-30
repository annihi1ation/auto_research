import os

from read_tex import extract_arxiv_ids_from_file
from trans_bibtex import save_arxiv_ids_as_bib


if __name__ == '__main__':

    paper_folder = './paper'
    tex_files = [f for f in os.listdir(paper_folder) if f.endswith('.tex')]  # 获取当前目录下所有 tex 文件（把 json 改掉了）

    for json_file in tex_files:  # 遍历每个 JSON 文件

        print('-'*100)
        print(f"文件名: {json_file}")  # 打印文件名
        arxiv_ids = extract_arxiv_ids_from_file(paper_folder, json_file)  # 调用函数提取编号
        print(f"提取到的编号: {arxiv_ids}")  # 打印提取到的编号数组

        save_arxiv_ids_as_bib(arxiv_ids, paper_folder, json_file)
