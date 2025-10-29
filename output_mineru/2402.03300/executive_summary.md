# Executive Summary

## Unlocking Mathematical Reasoning: How DeepSeekMath Bridges the Gap Between Open and Closed AI Models

### The Problem

Mathematical reasoning has long been one of the most challenging frontiers for artificial intelligence. While cutting-edge models like GPT-4 and Gemini-Ultra can solve complex math problems, these closed-source models remain inaccessible to researchers and developers. Open-source alternatives significantly trail behind in performance, creating a critical gap between what's possible in AI research and what's available to the broader community.

### The Breakthrough

DeepSeekMath 7B achieves **51.7% accuracy** on the competition-level MATH benchmark without external tools, approaching the performance of closed-source giants while being completely open. This breakthrough comes from two key innovations: a massive **120B-token mathematical corpus** harvested from publicly available web data using an engineered selection pipeline, and **Group Relative Policy Optimization (GRPO)**, a memory-efficient reinforcement learning algorithm that significantly enhances mathematical reasoning capabilities.

### How It Works

The researchers first built the DeepSeekMath Corpus by mining Common Crawl data using a sophisticated fastText-based classifier trained on OpenWebMath. They then continued pre-training DeepSeek-Coder-Base-v1.5 7B on this mathematical data, finding that starting from a code-trained model was superior to a general language model. Finally, they applied GRPO, which eliminates the need for a separate critic model by using group-based advantage estimation, reducing memory requirements while boosting performance. The result: **51.7% accuracy** on MATH (improving to **60.9%** with self-consistency) and **88.2%** on GSM8K.

![Figure 1 | Top1 accuracy of open-source models on the competition-level MATH benchmark](./images/cb99819eff702b4bd65201a60752efa7b69c62243765b427bd67b116d461a648.jpg)

### Why This Matters

This breakthrough democratizes advanced mathematical AI capabilities that were previously locked behind closed-source models. Researchers, educators, and developers can now access state-of-the-art mathematical reasoning abilities without relying on proprietary APIs. The multilingual nature of the training data also means improved performance across different languages, particularly in Chinese mathematical benchmarks.

### The Business Opportunity

The technology enables new applications in automated tutoring systems, mathematical research assistance, educational content generation, and STEM assessment tools. Organizations can build sophisticated mathematical reasoning capabilities without the high costs of licensing closed-source models or the computational expense of training massive models from scratch.