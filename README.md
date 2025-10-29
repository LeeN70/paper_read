# ArXiv Paper Reader

一个支持多种解析引擎（MinerU API 和 Zai API）的自动化 arXiv 论文处理工具，结合 Claude Agent SDK 将学术论文 PDF 自动解析并生成专业的总结报告。

## ✨ 核心功能

- 🔄 **双解析引擎支持**：灵活选择 MinerU 或 Zai 作为文档解析后端
  - **MinerU**：云端 API，直接从 URL 解析（默认）
  - **Zai**：本地化 API，支持更多定制选项
- 📄 **智能 PDF 解析**：自动提取论文内容、图表和公式
- 🤖 **AI 驱动摘要生成**：使用 Claude Agent SDK 生成两种层次的论文总结：
  - **执行摘要**（Executive Summary）：面向非技术读者的简洁概述
  - **详细分析**（Detailed Breakdown）：面向技术专家的深度剖析
- 🖼️ **图像自动提取**：自动从论文中提取和整理所有图表
- 📂 **结构化输出**：按论文 ID 和解析器类型组织的清晰目录结构
- 💾 **智能缓存**：自动缓存解析结果，避免重复处理

## 🚀 快速开始

### 安装依赖

1. 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

2. 确保已安装 Claude Code CLI（claude-agent-sdk 必需）：

```bash
npm install -g @anthropic-ai/claude-code
```

### 配置

在 `config.py` 中配置你的 API 设置：

```python
# MinerU API Token（如使用 MinerU）
MINERU_TOKEN = "your_api_token_here"

# Zai API 地址（如使用 Zai）
ZAI_BASE_URL = "http://10.243.65.197:12004"
```

### 使用方法

基本用法：

```bash
python main.py <arxiv_url> [--parser {mineru,zai}]
```

示例：

```bash
# 使用 MinerU 解析（默认）
python main.py https://arxiv.org/pdf/2502.17480.pdf

# 使用 MinerU 解析（显式指定）
python main.py https://arxiv.org/pdf/2502.17480.pdf --parser mineru

# 使用 Zai 解析
python main.py https://arxiv.org/pdf/1706.03762.pdf --parser zai
```

### 解析器选择

#### MinerU（默认）
- ✅ 云端服务，无需本地部署
- ✅ 直接从 URL 解析 PDF
- ✅ 支持公式识别（可选）
- ⚠️ 需要 API Token
- ⚠️ 有日限额（2000 页）

#### Zai
- ✅ 本地化部署，更可控
- ✅ 支持多种文档类型
- ✅ 可自定义 OCR 选项
- ⚠️ 需要先下载 PDF
- ⚠️ 需要访问内部服务

## 📁 项目结构

```
paper-reader-3/
├── main.py                  # CLI 入口，主流程控制
├── mineru_client.py         # MinerU API 客户端
├── zai_client.py            # Zai API 客户端
├── paper_processor.py       # Claude Agent SDK 处理器
├── md_to_pdf.py             # Markdown 转 PDF 工具
├── config.py                # 配置文件
├── requirements.txt         # Python 依赖
├── templates/               # 摘要模板
│   ├── executive_summary.md
│   └── detailed_breakdown.md
├── output_mineru/           # MinerU 生成的摘要输出
│   └── <paper_id>/
│       ├── executive_summary.md
│       ├── executive_summary.pdf
│       ├── detailed_breakdown.md
│       ├── detailed_breakdown.pdf
│       └── images/
├── output_zai/              # Zai 生成的摘要输出
│   └── <paper_id>/
│       ├── executive_summary.md
│       ├── executive_summary.pdf
│       ├── detailed_breakdown.md
│       ├── detailed_breakdown.pdf
│       └── images/
├── cache_mineru/            # MinerU 解析结果缓存
│   └── <paper_id>/
│       ├── full.md
│       └── images/
└── cache_zai/               # Zai 解析结果缓存
    └── <paper_id>/
        ├── res.md
        ├── layout.json
        └── imgs/
```

## 🔄 工作流程

### MinerU 工作流程

1. **URL 验证**：验证输入的 arXiv URL 格式
2. **提交解析任务**：将 PDF URL 提交到 MinerU API
3. **状态轮询**：每 10 秒检查一次解析进度，显示页数进度
4. **下载结果**：解析完成后下载并提取 ZIP 压缩包
5. **缓存内容**：将解析结果保存到 `cache_mineru/` 目录
6. **AI 分析**：Claude Agent 读取解析后的 Markdown 和图像
7. **生成摘要**：根据模板生成两份结构化的摘要报告
8. **输出保存**：将摘要和图像保存到 `output_mineru/` 目录

### Zai 工作流程

1. **URL 验证**：验证输入的 arXiv URL 格式
2. **下载 PDF**：从 arXiv 下载 PDF 到本地
3. **预上传**：获取上传 URL 和唯一 UID
4. **上传文件**：上传 PDF 文件到 Zai 服务
5. **触发解析**：提交异步解析任务
6. **状态轮询**：轮询解析状态直到完成
7. **下载结果**：下载并提取 tar 压缩包
8. **缓存内容**：将解析结果保存到 `cache_zai/` 目录
9. **AI 分析**：Claude Agent 读取解析后的 Markdown 和图像
10. **生成摘要**：根据模板生成两份结构化的摘要报告
11. **输出保存**：将摘要和图像保存到 `output_zai/` 目录

### 详细流程示例

```
ArXiv Paper Reader
======================================================================

Paper ID: 2502.17480

Step 1: Submitting to MinerU for PDF parsing
----------------------------------------------------------------------
Submitting task to MinerU for: https://arxiv.org/pdf/2502.17480.pdf
Task submitted successfully. Task ID: abc123...

Step 2: Waiting for MinerU to complete parsing
----------------------------------------------------------------------
Polling task status...
  Status: pending
  Status: running
  Progress: 5/15 pages
  Progress: 10/15 pages
  Progress: 15/15 pages
Task completed! Result URL: https://...

Step 3: Downloading and extracting parsed content
----------------------------------------------------------------------
Downloading result ZIP from: https://...
Extracting ZIP file...
Copying images to output directory...
Copied 8 images

Step 4: Generating summaries with Claude Agent SDK
----------------------------------------------------------------------
Starting Claude Agent to generate summaries...
This may take a few minutes...

Claude Agent finished processing.

======================================================================
SUCCESS!
======================================================================

Paper summaries generated successfully!

Output directory: /data/lixin/paper-reader-3/output/2502.17480
  - Executive Summary: output/2502.17480/executive_summary.md
  - Detailed Breakdown: output/2502.17480/detailed_breakdown.md
  - Images: output/2502.17480/images

Cached MinerU parsed content: cache/2502.17480
```

## 🎯 输出说明

### Executive Summary（执行摘要）

面向非技术读者的简洁概述，包括：
- 简洁有力的标题
- 研究要解决的核心问题
- 关键突破和创新点
- 工作原理的高层次解释
- 实际意义和影响
- 商业机会和应用前景

### Detailed Breakdown（详细分析）

面向技术专家的深度剖析，包括：
- 技术问题的详细说明
- 核心创新和技术突破
- 系统架构和实现细节
- 实验结果和关键指标
- 实际应用场景分析
- 局限性和注意事项
- 对开发者和建设者的启示

## ⚙️ 配置选项

在 `config.py` 中可以自定义：

```python
# MinerU API 配置
MINERU_TOKEN = "your_token"           # MinerU API 令牌
MINERU_BASE_URL = "https://mineru.net/api/v4"

# Zai API 配置
ZAI_BASE_URL = "http://10.243.65.197:12004"  # Zai 服务地址

# 轮询配置
POLL_INTERVAL_SECONDS = 10            # 状态检查间隔（秒）
TIMEOUT_SECONDS = 1800                # 超时时间（30 分钟）

# Claude SDK 配置
CLAUDE_ALLOWED_TOOLS = [              # 允许 Claude 使用的工具
    "Read", "Write", "Edit", "Grep", "Glob"
]
CLAUDE_PERMISSION_MODE = "acceptEdits"  # 权限模式

# 目录配置
TEMPLATES_DIR = "templates"           # 模板目录
CACHE_MINERU_DIR = "cache_mineru"     # MinerU 缓存目录
CACHE_ZAI_DIR = "cache_zai"           # Zai 缓存目录
OUTPUT_MINERU_DIR = "output_mineru"   # MinerU 输出目录
OUTPUT_ZAI_DIR = "output_zai"         # Zai 输出目录
```

## 📋 依赖要求

### Python 依赖

- Python 3.8+
- requests >= 2.31.0
- claude-agent-sdk >= 0.1.0
- pypandoc >= 1.11 (用于 Markdown 转 PDF)

### 外部工具

- Node.js（用于 Claude Code CLI）
- Claude Code CLI（`@anthropic-ai/claude-code`）
- Pandoc（用于 Markdown 转 PDF，可选）
- wkhtmltopdf（用于更好的 PDF 样式支持，可选）

## 🔍 解析引擎详细说明

### MinerU API

MinerU 是一个专业的云端 PDF 解析服务：

- **日限额**：2000 页（最高优先级）
- **文件限制**：最大 200MB，最多 600 页
- **解析时间**：通常 1-10 分钟（取决于论文长度）
- **输出格式**：ZIP 文件（包含 `full.md` 和 `images/` 目录）
- **支持功能**：
  - OCR 文字识别
  - 公式提取（可选）
  - 图表提取
  - 表格识别

### Zai API

Zai 是内部的文档解析服务：

- **部署方式**：本地化部署
- **解析时间**：取决于服务器负载和文档复杂度
- **输出格式**：tar 文件（包含 `res.md`、`layout.json` 和 `imgs/` 目录）
- **工作流程**：
  1. 预上传获取 URL
  2. 上传文件
  3. 触发异步解析
  4. 轮询结果
- **支持功能**：
  - OCR 文字识别（可选全文 OCR）
  - 布局检测
  - 公式检测
  - 表格结构识别

## 🛠️ 错误处理

工具内置了完善的错误处理机制：

- ✅ 无效 arXiv URL 检测
- ✅ MinerU API 错误和超时处理
- ✅ 文件缺失和解析失败检测
- ✅ Claude SDK 错误捕获
- ✅ 网络请求重试机制

## 💡 使用技巧

1. **首次使用**：处理一篇较短的论文（10 页以内）来测试配置
2. **解析器选择**：
   - MinerU：适合快速处理，无需本地资源
   - Zai：适合需要更多控制的场景，可自定义 OCR 参数
3. **批量处理**：可以编写脚本循环调用 `main.py` 处理多篇论文
4. **缓存利用**：
   - MinerU 缓存：`cache_mineru/<paper_id>/full.md`
   - Zai 缓存：`cache_zai/<paper_id>/res.md`
5. **模板定制**：修改 `templates/` 中的模板以适应特定需求
6. **图像引用**：生成的摘要会自动引用 `./images/` 中的图像
7. **对比测试**：可以用两种解析器处理同一论文，比较解析质量

## 📄 Markdown 转 PDF

项目提供了一个独立的脚本 `md_to_pdf.py`，可以将生成的摘要文档（Markdown 格式）批量转换为专业的 PDF 文档。

### 前置要求

1. **安装 Pandoc**：
   ```bash
   # Windows (使用 Chocolatey)
   choco install pandoc
   
   # macOS
   brew install pandoc
   
   # Linux (Ubuntu/Debian)
   sudo apt-get install pandoc
   
   # 或从官网下载：https://pandoc.org/installing.html
   ```

2. **安装 wkhtmltopdf（推荐，用于更好的样式支持）**：
   ```bash
   # Windows: 下载安装包
   # https://wkhtmltopdf.org/downloads.html
   
   # macOS
   brew install wkhtmltopdf
   
   # Linux (Ubuntu/Debian)
   sudo apt-get install wkhtmltopdf
   ```

3. **安装 Python 依赖**（如果还没有）：
   ```bash
   pip install pypandoc
   ```

### 使用方法

运行转换脚本，自动处理 `output_mineru/` 和 `output_zai/` 目录下的所有摘要文档：

```bash
python md_to_pdf.py
```

### 功能特性

- ✅ **自动扫描**：递归扫描所有子目录，找到 `executive_summary.md` 和 `detailed_breakdown.md`
- ✅ **图像嵌入**：自动解析 Markdown 中的图像路径并嵌入到 PDF
- ✅ **专业样式**：使用自定义 CSS 样式，提供专业的排版效果
  - 精美的标题和章节样式
  - 合理的边距和行距
  - 代码块语法高亮
  - 表格美化
  - 图片居中和边框
- ✅ **批量处理**：一次性处理所有目录中的文档
- ✅ **错误处理**：单个文件失败不影响其他文件的转换

### 输出位置

PDF 文件会保存在原 Markdown 文件的同一目录下：

```
output_mineru/2502.17480/
├── executive_summary.md
├── executive_summary.pdf      ← 新生成的 PDF
├── detailed_breakdown.md
├── detailed_breakdown.pdf     ← 新生成的 PDF
└── images/
    └── ...
```

### 转换示例输出

```
======================================================================
Markdown to PDF Converter
======================================================================

INFO: Pandoc version: 2.19.2
INFO: Scanning directories for markdown files...
INFO: Found 2 executive_summary.md file(s) in output_mineru
INFO: Found 2 detailed_breakdown.md file(s) in output_mineru
INFO: Found 1 executive_summary.md file(s) in output_zai
INFO: Found 1 detailed_breakdown.md file(s) in output_zai

INFO: Found 6 markdown file(s) to convert

[1/6] Converting executive_summary.md...
INFO: Processing: output_mineru/2502.17480/executive_summary.md
INFO: ✓ Successfully created: output_mineru/2502.17480/executive_summary.pdf

[2/6] Converting detailed_breakdown.md...
INFO: Processing: output_mineru/2502.17480/detailed_breakdown.md
INFO: ✓ Successfully created: output_mineru/2502.17480/detailed_breakdown.pdf

...

======================================================================
Conversion Summary
======================================================================
INFO: ✓ Successful: 6
INFO: Total: 6
======================================================================
```

## 🎨 自定义模板

模板使用 Markdown 格式，包含占位符和结构指南：

- `templates/executive_summary.md`：执行摘要模板
- `templates/detailed_breakdown.md`：详细分析模板

Claude Agent 会：
1. 读取论文内容和模板结构
2. 理解每个章节的要求
3. 从论文中提取相关信息
4. 按模板结构填充内容
5. 合理引用图表

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

---

**Note**: 
- 使用 MinerU 需要有效的 MinerU API Token
- 使用 Zai 需要访问内部 Zai 服务（`http://10.243.65.197:12004`）
- 两种解析器都需要 Claude API 访问权限来生成摘要
