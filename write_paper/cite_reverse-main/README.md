# 引用替换工具

这个工具用于将BibTeX文件中的引用条目替换为从DBLP获取的正式发表的论文引用。

## 功能特点

- 自动解析BibTeX文件中的所有引用条目
- 根据论文标题和作者在DBLP上搜索对应的正式发表版本
- 保留原始引用标识符(cite_key)，仅替换内容
- 智能控制请求频率，避免被DBLP限制访问
- 对未找到匹配的条目添加警告注释并保留原始条目
- 智能识别DBLP中仍是arXiv预印本的文献，保留原条目并添加注释说明

## 安装依赖

```bash
pip install requests beautifulsoup4
```

## 使用方法

默认使用：
```bash
python cite_reverse_dblp.py
```
默认读取`cite.bib`文件并生成`cite_dblp.bib`作为输出。

自定义输入输出文件：
```python
if __name__ == "__main__":
    main(input_file="你的输入文件.bib", output_file="你的输出文件.bib")
```

## 示例

原始条目：
```bibtex
@misc{1401.0514,
  author = {Chris J. Maddison and Daniel Tarlow},
  title = {Structured Generative Models of Natural Source Code},
  year = {2014},
  primaryClass = {cs.PL cs.LG stat.ML},
  url = {https://arxiv.org/abs/1401.0514}
}
```

替换后的条目：
```bibtex
@inproceedings{1401.0514,
  author       = {Chris J. Maddison and
                  Daniel Tarlow},
  title        = {Structured Generative Models of Natural Source Code},
  booktitle    = {Proceedings of the 31th International Conference on Machine Learning,
                  {ICML} 2014, Beijing, China, 21-26 June 2014},
  series       = {{JMLR} Workshop and Conference Proceedings},
  volume       = {32},
  pages        = {649--657},
  publisher    = {JMLR.org},
  year         = {2014},
  url          = {http://proceedings.mlr.press/v32/maddison14.html}
}
```

## 注意事项

- 程序可能不总是找到完全匹配的条目，建议手动检查替换结果
- 处理大型BibTeX文件可能需要较长时间，因为需要限制请求频率
- 如果频繁使用可能会被DBLP暂时限制访问，此时需等待一段时间后再试
- 对于DBLP中仍是预印本的论文，程序会保留原始条目并添加说明注释 