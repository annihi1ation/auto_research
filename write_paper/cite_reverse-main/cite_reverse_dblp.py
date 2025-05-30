#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import time
import urllib.parse
import requests
from bs4 import BeautifulSoup
import random
import argparse


# 不同的 User-Agent 列表，模拟不同的浏览器
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36 Edg/96.0.1054.29",
]

# 请求间隔控制（秒）
MIN_REQUEST_INTERVAL = 5  # 最小间隔时间，这个已经很快了，避免封号，别改这里
MAX_REQUEST_INTERVAL = 60  # 最大间隔时间
CURRENT_INTERVAL = MIN_REQUEST_INTERVAL  # 当前间隔时间
last_request_time = 0  # 上次请求时间


def parse_bib_file(file_path):

    """解析 BibTeX 文件，返回一个条目列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 正则表达式匹配 BibTeX 条目
    pattern = r"(@\w+\{[^@]+\})"
    entries = re.findall(pattern, content, re.DOTALL)
    return entries


def extract_entry_info(entry):

    """从 BibTeX 条目中提取关键信息"""
    # 提取条目类型和引用标识符
    type_key_match = re.match(r"@(\w+)\{([^,]+),", entry)
    if not type_key_match:
        return None

    entry_type, cite_key = type_key_match.groups()

    # 提取作者
    author_match = re.search(r"author\s*=\s*\{([^}]+)\}", entry)
    author = author_match.group(1) if author_match else ""

    # 提取标题
    title_match = re.search(r"title\s*=\s*\{([^}]+)\}", entry)
    title = title_match.group(1) if title_match else ""
    # 清理多行标题
    title = re.sub(r"\s+", " ", title)

    # 提取 URL
    url_match = re.search(r"url\s*=\s*\{([^}]+)\}", entry)
    url = url_match.group(1) if url_match else ""

    return {
        "entry_type": entry_type,
        "cite_key": cite_key,
        "author": author,
        "title": title,
        "url": url,
        "full_entry": entry,
    }


def make_request(url, timeout=30):

    """执行 HTTP 请求的函数，包含请求频率控制"""
    global CURRENT_INTERVAL, last_request_time

    # 计算需要等待的时间
    now = time.time()
    # 随机化等待时间，避免固定模式，增加了随机范围
    wait_time = max(
        0, last_request_time + CURRENT_INTERVAL + random.uniform(-1, 3) - now
    )

    if wait_time > 0:
        print(f"  等待 {wait_time:.2f} 秒...")
        time.sleep(wait_time)

    # 随机选择一个 User-Agent
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # 请求成功，可以适当减少间隔时间
        CURRENT_INTERVAL = max(MIN_REQUEST_INTERVAL, CURRENT_INTERVAL * 0.95)
        last_request_time = time.time()

        return response

    except requests.exceptions.HTTPError as e:
        # 请求失败，增加间隔时间
        if e.response.status_code == 429 or "Too Many Requests" in str(e):
            CURRENT_INTERVAL = min(MAX_REQUEST_INTERVAL, CURRENT_INTERVAL * 1.5)
            print(f"  遇到限流，增加请求间隔至 {CURRENT_INTERVAL:.2f} 秒")
        last_request_time = time.time()
        raise


def search_dblp(author, title):

    """在 DBLP 上搜索论文，返回 BibTeX 条目"""
    # 构建搜索查询
    query = f"{title} {author.split(' and ')[0]}"  # 使用标题和第一作者
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://dblp.org/search?q={encoded_query}"

    try:
        # 发送请求
        print(f"  正在搜索DBLP: {search_url}")
        print(f"  当前请求间隔: {CURRENT_INTERVAL:.2f} 秒")

        response = make_request(search_url)

        # 解析 HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # 查找搜索结果
        result_items = soup.select(".publ-list .entry")
        if not result_items:
            print("  未找到搜索结果")
            return None

        print(f"  找到 {len(result_items)} 个搜索结果")

        # 获取第一个结果的 BibTeX 链接
        for item in result_items:
            # 检查是否存在 BibTeX 链接
            bibtex_link = item.select_one('nav.publ a[href*="bibtex"]')
            if bibtex_link:
                bibtex_url = bibtex_link["href"]
                # 确保 URL 是完整的
                if not bibtex_url.startswith("http"):
                    if bibtex_url.startswith("/"):
                        bibtex_url = f"https://dblp.org{bibtex_url}"
                    else:
                        bibtex_url = f"https://dblp.org/{bibtex_url}"

                print(f"  获取BibTeX: {bibtex_url}")

                # 获取 BibTeX 内容
                bibtex_response = make_request(bibtex_url)
                bibtex_soup = BeautifulSoup(bibtex_response.text, "html.parser")
                bibtex_content = bibtex_soup.select_one("#bibtex-section pre")

                if bibtex_content:
                    return bibtex_content.text

        print("  未找到BibTeX内容")
        return None

    except Exception as e:
        print(f"  搜索DBLP时出错: {e}")
        return None


def replace_entry_with_dblp(arxiv_entry, dblp_entry):

    """
    用 DBLP 条目替换 arXiv 条目，但保留原始的 cite_key
    """
    if not dblp_entry:
        return arxiv_entry

    # 提取 arXiv 条目的 cite_key
    arxiv_info = extract_entry_info(arxiv_entry)
    if not arxiv_info:
        return arxiv_entry

    cite_key = arxiv_info["cite_key"]

    # 替换 DBLP 条目中的 cite_key
    dblp_entry_new = re.sub(r"@(\w+)\{([^,]+),", f"@\\1{{{cite_key},", dblp_entry)

    return dblp_entry_new


def process_entry(entry, output_file):

    """处理单个引用条目的函数"""
    info = extract_entry_info(entry)

    if not info:
        with open(output_file, "a", encoding="utf-8") as f_out:
            f_out.write(entry + "\n\n")
        return

    # 处理所有条目，无论是否为 arXiv
    print(f"处理条目: {info['cite_key']}")

    # 搜索 DBLP
    dblp_entry = search_dblp(info["author"], info["title"])

    with open(output_file, "a", encoding="utf-8") as f_out:
        if dblp_entry:
            # 检查 DBLP 返回的条目是否包含 eprinttype = {arXiv}
            is_arxiv_in_dblp = re.search(
                r"eprinttype\s*=\s*\{\s*arXiv\s*\}", dblp_entry, re.IGNORECASE
            )

            if is_arxiv_in_dblp:
                print(f"  找到DBLP匹配，但仍是arXiv预印本: 保留原条目并添加注释")
                # comment = f"% 在DBLP中找到匹配的条目，但该文献在DBLP上也是预印本，未正式发表，所以保留原条目。\n"
                comment = ''
                f_out.write(comment + entry + "\n\n")
            else:
                print(f"  找到DBLP匹配: 替换条目")
                new_entry = replace_entry_with_dblp(entry, dblp_entry)
                f_out.write(new_entry + "\n\n")
        else:
            print(f"  未找到DBLP匹配: 保留原条目并添加注释")
            # 添加注释说明未找到匹配
            # comment = f"% WARNING: 未在DBLP中找到匹配的条目: {info['cite_key']}。建议人工重新检查。\n"
            comment = ''
            f_out.write(comment + entry + "\n\n")


def main(input_file, output_file):

    print(f"开始处理文件 {input_file}...")

    # 解析 BibTeX 文件
    entries = parse_bib_file(input_file)
    print(f"找到 {len(entries)} 条引用条目")

    # 创建新文件（清空原有内容）
    with open(output_file, "w", encoding="utf-8"):
        pass

    # 顺序处理条目
    for i, entry in enumerate(entries):
        print(f"处理条目 {i+1}/{len(entries)}")
        try:
            process_entry(entry, output_file)
        except Exception as e:
            print(f"处理条目时出错: {e}")
            # 出错时增加延迟
            time.sleep(random.uniform(10, 20))

    print(f"处理完成。新文件已保存为 {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='将arXiv引用替换为DBLP引用')
    parser.add_argument('--input', '-i', default="cite.bib", help='输入BibTeX文件路径')
    parser.add_argument('--output', '-o', default="cite_dblp.bib", help='输出BibTeX文件路径')

    args = parser.parse_args()

    main(input_file=args.input, output_file=args.output)
    input('\n>> 已完成！')
