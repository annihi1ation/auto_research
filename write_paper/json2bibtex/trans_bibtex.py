import os
import json
from tqdm import tqdm


def save_arxiv_ids_as_bib(arxiv_ids, paper_folder, tex_file):

    # 输出文件路径
    output_file = os.path.join(paper_folder, os.path.basename(tex_file).replace('.tex', '.bib'))

    # 打开 JSON 文件并逐行解析
    with open('json2bibtex/data/arxiv-metadata-oai-snapshot.json', 'r', encoding='utf-8') as f:

        bib_entries = []  # 用于存储 BibTeX 条目
        for line in tqdm(f, ncols=150, desc='bibtex'):
            metadata = json.loads(line.strip())  # 解析每一行 JSON 数据

            # 检查当前条目的 ID 是否在目标 ID 列表中
            if metadata['id'] in arxiv_ids:

                year = int(metadata['id'][:2])
                year = 2000 + year if year < 50 else 1900 + year

                authors = ' and '.join([f"{author[1]} {author[0]}" for author in metadata['authors_parsed']])
                bib_entry = f"""@misc{{{metadata['id']},
  author = {{{authors}}},
  title = {{{metadata['title'].strip()}}},
  year = {{{str(year)}}},
  journal={{arXiv e-prints}},
  primaryClass = {{{metadata['categories']}}},
  url = {{https://arxiv.org/abs/{metadata['id']}}}
}}"""
                bib_entries.append(bib_entry)

    # 将 BibTeX 条目写入输出文件
    with open(output_file, 'w', encoding='utf-8') as bib_file:
        bib_file.write('\n\n'.join(bib_entries))

    # print(f"保存 bib: {output_file}")


if __name__ == "__main__":

    pass
