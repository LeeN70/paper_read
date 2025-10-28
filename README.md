# ArXiv Paper Reader

一个基于 MinerU API 和 Claude Agent SDK 的自动化 arXiv 论文处理工具，能够将学术论文 PDF 自动解析并生成专业的总结报告。

## ✨ 核心功能

- 📄 **智能 PDF 解析**：通过 MinerU API 自动提取论文内容、图表和公式
- 🤖 **AI 驱动摘要生成**：使用 Claude Agent SDK 生成两种层次的论文总结：
  - **执行摘要**（Executive Summary）：面向非技术读者的简洁概述
  - **详细分析**（Detailed Breakdown）：面向技术专家的深度剖析
- 🖼️ **图像自动提取**：自动从论文中提取和整理所有图表
- 📂 **结构化输出**：按论文 ID 组织的清晰目录结构
- 💾 **智能缓存**：自动缓存 MinerU 解析结果，避免重复处理

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

在 `config.py` 中配置你的 MinerU API Token：

```python
MINERU_TOKEN = "your_api_token_here"
```

### 使用方法

基本用法：

```bash
python main.py <arxiv_url>
```

示例：

```bash
python main.py https://arxiv.org/pdf/2502.17480.pdf
```

## 📁 项目结构

```
paper-reader-3/
├── main.py                  # CLI 入口，主流程控制
├── mineru_client.py         # MinerU API 客户端
├── paper_processor.py       # Claude Agent SDK 处理器
├── config.py                # 配置文件
├── requirements.txt         # Python 依赖
├── templates/               # 摘要模板
│   ├── executive_summary.md
│   └── detailed_breakdown.md
├── output/                  # 生成的摘要输出
│   └── <paper_id>/
│       ├── executive_summary.md
│       ├── detailed_breakdown.md
│       └── images/
└── cache/                   # MinerU 解析结果缓存
    └── <paper_id>/
        ├── full.md
        └── images/
```

## 🔄 工作流程

1. **URL 验证**：验证输入的 arXiv URL 格式
2. **提交解析任务**：将 PDF URL 提交到 MinerU API
3. **状态轮询**：每 10 秒检查一次解析进度，显示页数进度
4. **下载结果**：解析完成后下载并提取 ZIP 压缩包
5. **缓存内容**：将 MinerU 的解析结果保存到 `cache/` 目录
6. **AI 分析**：Claude Agent 读取解析后的 Markdown 和图像
7. **生成摘要**：根据模板生成两份结构化的摘要报告
8. **输出保存**：将摘要和图像保存到 `output/` 目录

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

# 轮询配置
POLL_INTERVAL_SECONDS = 10            # 状态检查间隔（秒）
TIMEOUT_SECONDS = 1800                # 超时时间（30 分钟）

# Claude SDK 配置
CLAUDE_ALLOWED_TOOLS = [              # 允许 Claude 使用的工具
    "Read", "Write", "Edit", "Grep", "Glob"
]
CLAUDE_PERMISSION_MODE = "acceptEdits"  # 权限模式

# 目录配置
OUTPUT_DIR = "output"                 # 输出目录
TEMPLATES_DIR = "templates"           # 模板目录
CACHE_DIR = "cache"                   # 缓存目录
```

## 📋 依赖要求

### Python 依赖

- Python 3.8+
- requests >= 2.31.0
- claude-agent-sdk >= 0.1.0

### 外部工具

- Node.js（用于 Claude Code CLI）
- Claude Code CLI（`@anthropic-ai/claude-code`）

## 🔍 MinerU API 说明

MinerU 是一个专业的 PDF 解析服务：

- **日限额**：2000 页（最高优先级）
- **文件限制**：最大 200MB，最多 600 页
- **解析时间**：通常 1-10 分钟（取决于论文长度）
- **支持功能**：
  - OCR 文字识别
  - 公式提取（可选）
  - 图表提取
  - 表格识别

## 🛠️ 错误处理

工具内置了完善的错误处理机制：

- ✅ 无效 arXiv URL 检测
- ✅ MinerU API 错误和超时处理
- ✅ 文件缺失和解析失败检测
- ✅ Claude SDK 错误捕获
- ✅ 网络请求重试机制

## 💡 使用技巧

1. **首次使用**：处理一篇较短的论文（10 页以内）来测试配置
2. **批量处理**：可以编写脚本循环调用 `main.py` 处理多篇论文
3. **缓存利用**：如果只想重新生成摘要，直接使用 `cache/` 中的内容
4. **模板定制**：修改 `templates/` 中的模板以适应特定需求
5. **图像引用**：生成的摘要会自动引用 `./images/` 中的图像

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

**Note**: 本工具需要有效的 MinerU API Token 和 Claude API 访问权限才能正常工作。
