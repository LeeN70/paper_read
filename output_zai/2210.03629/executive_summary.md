# Executive Summary

## ReAct: Synergizing Reasoning and Acting in Language Models

### The Problem

Large language models have traditionally been studied for either reasoning capabilities (like chain-of-thought prompting) or action generation (like interacting with environments), but not both together. This separation limits their effectiveness - reasoning-only approaches can "hallucinate" facts and suffer from error propagation, while action-only approaches lack the strategic planning and abstract thinking needed for complex tasks.

### The Breakthrough

ReAct introduces a novel paradigm that combines **reasoning traces** and **task-specific actions** in an interleaved manner, allowing language models to both think through problems and interact with external sources like Wikipedia or web environments. This synergy enables models to create, maintain, and adjust high-level plans while gathering real-world information to support their reasoning.

### How It Works

ReAct prompts language models to generate verbal reasoning traces (thoughts) and actions in sequence, creating thought-action-observation cycles. The model might decompose a problem, search for information, analyze the results, and then decide on next steps - all while maintaining a working memory of the process. On interactive decision-making tasks, ReAct achieved **34% absolute improvement** over imitation learning methods on ALFWorld and **10% improvement** on WebShop, using only 1-2 examples compared to thousands of training instances for baselines.

### Why This Matters

This approach significantly reduces fact hallucination and error propagation while creating more interpretable and trustworthy AI systems. The combination of internal reasoning and external information gathering makes AI responses more factual and grounded, while the reasoning traces make it easier for humans to understand and verify the model's decision-making process.

### The Business Opportunity

ReAct enables more reliable and capable AI assistants for knowledge-intensive tasks like research, customer service, and complex problem-solving, while also opening doors for more sophisticated autonomous agents that can both think strategically and take effective action in real-world environments.