# Executive Summary

## DeepSeek-R1: Teaching AI Models to Think Through Reinforcement Learning

### The Problem

Large language models have struggled with complex reasoning tasks like mathematics, coding, and scientific problem-solving. Traditional approaches require massive amounts of human-supervised training data, which is expensive and time-consuming to create. While OpenAI's o1 models showed promising results by using longer "thinking time" during inference, the research community lacked open alternatives that could achieve comparable performance.

### The Breakthrough

DeepSeek-R1 introduces a revolutionary approach: using **pure reinforcement learning (RL)** to teach language models how to reason without relying on supervised fine-tuning as a first step. The researchers created two models—DeepSeek-R1-Zero (pure RL) and DeepSeek-R1 (RL with minimal cold-start data)—that spontaneously develop sophisticated reasoning behaviors like self-reflection and exploring alternative solutions, similar to how humans solve complex problems.

### How It Works

The system uses a clever RL algorithm called GRPO that optimizes models by having them generate multiple solutions to each problem and learn from the outcomes. DeepSeek-R1-Zero starts with a base model and, through thousands of RL training steps, naturally learns to spend more time thinking through problems—generating hundreds to thousands of reasoning tokens per answer. Remarkably, the model experiences an **"aha moment"** where it learns to stop and rethink its approach when it detects errors. The enhanced DeepSeek-R1 adds a small amount of human-friendly cold-start data to improve readability while maintaining the powerful reasoning capabilities.

### Why This Matters

This research democratizes advanced reasoning capabilities that were previously locked behind proprietary models like OpenAI's o1. DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on mathematics (79.8% vs 79.2% on AIME 2024) and coding tasks (2029 vs 2061 Codeforces rating), while being open-source. The approach also works for smaller models—through distillation, a 14B parameter model outperforms the much larger 32B QwQ-32B-Preview, making powerful reasoning more accessible and efficient.

### The Business Opportunity

Organizations can now deploy sophisticated reasoning models without relying on expensive proprietary APIs. The combination of open-source availability and efficient distillation means companies can run powerful reasoning models on their own infrastructure, reducing costs and increasing privacy. This opens up applications in education, scientific research, software development, and any field requiring complex problem-solving capabilities.