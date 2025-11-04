#!/bin/bash

# 批处理脚本：依次处理多个arXiv论文

echo "开始批处理论文..."

# 第一组
echo "正在处理: 1706.03762.pdf"
python main.py https://arxiv.org/pdf/1706.03762.pdf --parser zai

echo "正在处理: 1810.04805.pdf"
python main.py https://arxiv.org/pdf/1810.04805.pdf --parser zai

echo "正在处理: 2203.02155.pdf"
python main.py https://arxiv.org/pdf/2203.02155.pdf --parser zai

# 第二组
echo "正在处理: 2501.12948.pdf"
python main.py https://arxiv.org/pdf/2501.12948.pdf --parser zai

echo "正在处理: 2507.20534.pdf"
python main.py https://arxiv.org/pdf/2507.20534.pdf --parser zai

echo "正在处理: 2508.06471.pdf"
python main.py https://arxiv.org/pdf/2508.06471.pdf --parser zai

# 第三组
echo "正在处理: 2210.03629.pdf"
python main.py https://arxiv.org/pdf/2210.03629.pdf --parser zai

echo "批处理完成！"

