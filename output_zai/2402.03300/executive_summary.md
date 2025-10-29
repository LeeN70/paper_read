# Executive Summary

## DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

### The Problem

Mathematical reasoning poses a significant challenge for language models due to its complexity and structured nature. While cutting-edge models like GPT-4 and Gemini-Ultra demonstrate impressive mathematical capabilities, they remain closed-source and inaccessible to the research community. Open-source models substantially trail behind in performance, leaving a critical gap in mathematical AI that limits scientific progress and practical applications.

### The Breakthrough

DeepSeekMath 7B achieves **51.7% accuracy** on the competition-level MATH benchmark without external tools, approaching the performance of Gemini-Ultra and GPT-4. This breakthrough comes from two key innovations: first, a meticulously engineered data pipeline that extracts 120B high-quality math tokens from publicly available Common Crawl data, and second, **Group Relative Policy Optimization (GRPO)**, an efficient reinforcement learning algorithm that significantly enhances mathematical reasoning while reducing memory usage.

### How It Works

The system starts with DeepSeek-Coder-Base-v1.5 7B and continues pretraining on a specially curated mathematical corpus. The key insight is that publicly available web data contains valuable mathematical content when properly filtered. GRPO eliminates the need for a separate critic model by using group-based baseline estimation, making reinforcement learning **60% more memory-efficient**. With just 64 samples, DeepSeekMath achieves **60.9%** accuracy on MATH through self-consistency.

### Why This Matters

This development democratizes advanced mathematical reasoning capabilities, making them accessible to researchers, educators, and developers who previously relied on closed-source models. The model's strong performance across both English and Chinese mathematical benchmarks opens up possibilities for global educational applications and scientific computing tools that were previously limited to proprietary systems.

### The Business Opportunity

The efficient architecture and open-source nature create opportunities for companies to build specialized mathematical AI applications without the massive computational costs typically required. From automated tutoring systems to advanced research assistants, organizations can now deploy sophisticated mathematical reasoning capabilities at a fraction of previous costs.