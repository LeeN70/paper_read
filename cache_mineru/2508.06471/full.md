# GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models

# GLM-4.5 Team

Zhipu AI & Tsinghua University

(For the complete list of authors, please refer to the Contribution section)

# Abstract

We present GLM-4.5, an open-source Mixture-of-Experts (MoE) large language model with 355B total parameters and 32B activated parameters, featuring a hybrid reasoning method that supports both thinking and direct response modes. Through multi-stage training on 23T tokens and comprehensive post-traiing with expert model iteration and reinforcement learning, GLM-4.5 achieves strong performance across agentic, reasoning,and coding (ARC) tasks, scoring 7O.1% on TAU-Bench,91.0% on AIME 24,and 64.2% on SWE-bench Verified.With much fewer parameters than several competitors, GLM-4.5 ranks 3rd overall among all evaluated models and 2nd on agentic benchmarks.We release both GLM-4.5 (355B parameters)and a compact version, GLM-4.5-Air (106B parameters), to advance research in reasoning and agentic AI systems. Code, models,and more information are available at https://github.com/zai-org/GLM-4.5.

![](images/1c188624ca02a86a6173228469f50949b84834fde8ce97ffa41c4e7b3c559759.jpg)  
LLM Performance Evaluation: Agentic, Reasoning,and Coding Benchmarks   
12enchmarks:-o24,50,ode,,HEB-5)EBeefdaBchBch,o   
Figure 1: Average performance on agentic, reasoning,and coding (ARC) benchmarks. Overall, GLM-4.5 achieves a rank of 3rd, with GLM-4.5-Air following at rank 6th. The models listed are evaluated as of July 28, 2025.

# 1Introduction

Large language models (LLMs) are rapidly evolving from general knowledge repositories [6; 37; 50; 33; 23] into general problem-solvers.The ultimate ambition,often associated with Artificial General Intelligence (AGI), is to create models with human-level cognitive capabilities across diverse domains. This requires a unified mastery of complex problem-solving, generalization,and self-improvement, moving beyond task-specific excellence.

As LLMs become more integrated into real-world scenarios, the key to enhancing actual productivity and solving complex professional tasks lies in developing specific core capabilities. We identify three critical, interconnected capabilities as the measure of a truly generalist model: Agentic abilities for interacting with external tools and the real world; complex Reasoning for solving multi-step problems in domains like mathematics and science; and advanced Coding skills for tackling realworld software engineering tasks. While state-of-the-art proprietary models like OpenAI's ol/o3 [18] and Anthropic's Claude Sonnet 4 have demonstrated groundbreaking performance in specific ARC domains (e.g., mathematical reasoning or code fixing [2Ol), a single, powerful open-source model that excels across all three areas has remained elusive.

This paper introduces two new models: GLM-4.5 and GLM-4.5-Air, toward the goal of unifying all the different capabilities. The new models outperform existing open-source LLM models [13; 34; 47] across the board, with significant gains in agentic,reasoning,and coding tasks. GLM-4.5 and GLM4.5-Air both feature hybrid reasoning modes: thinking mode for complex reasoning and agentic tasks,and non-thinking mode for instant responses. GLM-4.5 is our first MoE model, with 355B total parameters and 32B activated parameters. GLM-4.5 demonstrates strong performance on the following ARC benchmarks:

· Agentic: GLM-4.5 scores 70.1% on TAU-Bench and 77.8% on BFCL v3 [26], on par with Claude Sonnet 4. For web browsing agents, GLM-4.5 scores 26.4% on BrowseComp [45], clearly outperforming Claude Opus 4 (18.8%) and close to o4-mini-high (28.3%).   
· Reasoning: GLM-4.5 demonstrates outstanding performance on a suite of challnging reasoning benchmarks,achieving 91.0% on AIME 24,79.1% on GPQA [30], 72.9% on LiveCodeBench (2407-2501) [19],and 14.4% on HLE (Humanity's Last Exam) [28].   
· Coding: GLM-4.5 scores 64.2% on SWE-bench Verified [20] and 37.5% on Terminal-Bench [35], outperforming GPT-4.1 and Gemini-2.5-pro, close to Claude Sonnet 4.

GLM-4.5-Air is a smaller MoE model with 106B parameters. It represents a significant leap among models at the 100B scale, matching or exceeding Qwen3-235B-A22B [47] and MiniMax-M1 [7].

In Figure 1, we show the average performance on 12 benchmarks across agentic,reasoning, and coding (ARC) tasks. Overall, GLM-4.5 is ranked in the 3rd place and GLM-4.5-Air is ranked in the 6th. On agentic tasks, GLM-4.5 is ranked in the 2nd place, following OpenAI o3. On coding tasks, GLM-4.5 is ranked in the third place,close to Claude Sonnet 4. Note that GLM-4.5 is highly parameter-efficient, with only half the parameters of DeepSeek-R1 [13] and one-third those of Kimi K2 [34]. In Figure 2, we report the scores on SWE-bench Verified vs model parameters of different open-source models, where GLM-4.5 and GLM-4.5-Air lie on the Pareto Frontier. More evaluation results are detailed in Section 4.

Both GLM-4.5 and GLM-4.5-Air are available on Z.ai, BigModel.cn,and also as open-source models on https://huggingface.co/zai-org/GLM-4.5. We also open-sourced an evaluation toolkit at https: //github.com/zai-org/glm-simple-evals to ensure the reproducibility of our benchmark results.

# 2Pre-Training

# 2.1 Architecture

In the GLM-4.5 series, we adopt the MoE architecture, which improves the computational efficiency of both training and inference. We employ loss-free balance routing [40] and sigmoid gates for MoE layers [23]. Different from DeepSeek-V3 [23]and Kimi K2 [34], we reduce the width (hidden dimension and number of routed experts)of the model and increase its height (number of layers),as we found that deeper models exhibited better reasoning capacity. In the self-attention component, we employ Grouped-Query Attention with partial RoPE.Furthermore, we utilize 2.5 times more attention heads (96 heads for a 512O hidden dimension). Counterintuitively, while this increased head count does not improve training loss compared to models with fewer heads, it consistently improves performance on reasoning benchmarks such as MMLU and BBH. We also incorporate QK-Norm [15] to stabilize the range of attention logits. For both GLM-4.5 and GLM-4.5-Air, we add an MoE layer as the MTP (Multi-Token Prediction) layer [12] to support speculative decoding during inference.

![](images/fcf2722168b874740305ddcefdc2672d3770f4a4f542c72bb7287324fd514511.jpg)  
Figure 2: SWE-bench verified scores vs model parameters. Proprietary models are listed as unknown at the right side.

Table 1: Model architecture of GLM-4.5 and GLM-4.5-Air. When counting parameters,for GLM-4.5 and GLM-4.5-Air, we include the parameters of MTP layers but not word embeddings and the output layer.   

<table><tr><td>Model</td><td>GLM-4.5</td><td>GLM-4.5-Air</td><td>DeepSeek-V3</td><td>Kimi K2</td></tr><tr><td># Total Parameters</td><td>355B</td><td>106B</td><td>671B</td><td>1043B</td></tr><tr><td>#ActivatedParameters</td><td>32B</td><td>12B</td><td>37B</td><td>32B</td></tr><tr><td># Dense Layers</td><td>3</td><td>1</td><td>3</td><td>1</td></tr><tr><td>#MoE Layers</td><td>89</td><td>45</td><td>58</td><td>60</td></tr><tr><td># MTPLayers</td><td>1</td><td>1</td><td>1</td><td>0</td></tr><tr><td>Hidden Dim</td><td>5120</td><td>4096</td><td>7168</td><td>7168</td></tr><tr><td>Dense Intermediate Dim</td><td>12288</td><td>10944</td><td>18432</td><td>18432</td></tr><tr><td>MoE Intermediate Dim</td><td>1536</td><td>1408</td><td>2048</td><td>2048</td></tr><tr><td>Attention Head Dim</td><td>128</td><td>128</td><td>192</td><td>192</td></tr><tr><td># Attention Heads</td><td>96</td><td>96</td><td>128</td><td>64</td></tr><tr><td>#Key-Value Heads</td><td>8</td><td>8</td><td>128</td><td>64</td></tr><tr><td># Experts (total)</td><td>160</td><td>128</td><td>256</td><td>384</td></tr><tr><td># Experts Active Per Token</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td># Shared Experts</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>QK-Norm</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr></table>

# 2.2Pre-Training Data

Our pre-training corpus includes documents from webpages,social media, books, papers,and code repositories.We carefully design the data processing pipelines for different sources.

WebThe majority of our pre-training documents are English and Chinese webpages crawled from the Internet. Inspired by Nemotron-CC [32], we divide the crawled webpages into buckets of different quality scores. We up-sample documents from the bucket with higher quality scores and discard documents from the bucket with the lowest quality scores. The bucket with the highest quality scores contributes over 3.2 epochs during pre-training. In this way, the pre-training corpus can emphasize the high-frequency knowledge for reasoning tasks and also improve coverage for long-tail world knowledge. We have also found a large number of similar webpages automatically generated from templates and assigned high scores. Such webpages cannot be removed by MinHash deduplication. We additionally apply the SemDedup [1] pipeline to remove those similar webpages based on document embeddings.

![](images/00e56dd5195760a6ad4b4cc0c45d9111d5f4626451c47d32498e277a637e2911.jpg)  
Figure 3: Pre-training and mid-training stages for GLM-4.5. We adapt a multi-stage training recipe and extend the sequence length from 4K to 128K.

MultilingualTo support more natural languages, we include multilingual documents in our pretraining corpus. The multilingual corpus comes from both our crawled webpages and Fineweb-2 [27]. We apply a quality clasifier that judges the educational utility of documents and up-sample highquality multilingual documents.

CodeWe curated source code data from GitHub and various code hosting platforms. The code corpus undergoes a preliminary rule-based filtering,followed by classification using language-specific quality models that categorize samples into three tiers: high-quality, medium-quality,and low-quality. During training, we up-sampled high-quality code while excluding low-quality samples. Moreover, the Fill-In-the-Middle [5] training objective is applied to al source code data.For code-related web documents, we employ a two-stage retrieval process from our text pre-training corpus. Documents are initially selected based on two criteria: presence of HTML code tags, or identification by a FastText [22] classifier trained to detect code-related content. Subsequently,the retrieved documents undergo quality assessment using a dedicated model that classifies them into high-, medium-, or low-quality categories, following the same quality-based sampling strategy for source code. Finally, a fine-grained parser is employed to re-parse the selected web pages to better preserve the formats and contents of the code.

Math & ScienceTo enhance the reasoning capacity, we collct documents related to mathematics and science from webpages, books,and papers.We apply a large language model to score candidate documents based on the ratio of educational content about mathematics and science,and train a small-scale classifer to predict the scores. Documents in the pre-training corpus with scores above a certain threshold are up-sampled.

The pre-training process of GLM-4.5 is divided into two stages. In the first stage,the model is mainly trained on general documents from webpages. During the second stage, we up-sample the source code from GitHub and webpages related to coding,mathematics,and science.

# 2.3Mid-Training: Boost Reasoning & Agentic Capacity

After pre-training, we add several stages to further boost the model's performance on important application areas. Unlike traditional pre-training on large-scale general documents, these training stages utilize medium-size domain-specific datasets,including instruction data. Therefore, we denote these training stages as mid-training, which includes the following.

Repo-level Code TrainingAt this training stage, we add concatenated code files from the same repository to learn cross-file dependency. To improve the model's software engineering capability we also include model-filtered issues, pul requests (PRs),and commits from GitHub, with related issues,PRs,and commits concatenated into one context and commits organized in a dif-like format. We extend the training sequence length from 4K to 32K to incorporate large repositories.

Synthetic Reasoning Data TrainingAt this stage, we add synthetic reasoning content for math, science,and coding competitions. We collect a large number of questions and answers related to the reasoning tasks from webpages and books,and synthesize reasoning processes with a reasoning model.

Long-context & Agent TrainingTo further boost the model's long-context performance, we extend the training sequence length from 32K to 128K and up-sample long documents from the pre-training corpus. Large-scale synthetic agent trajectories are also incorporated at this stage.

In Figure 3, we show the complete stages for pre-training and mid-training. The maximum sequence length is kept at 4,096 in pre-training,and is extended from 32,768 to 131,072 in mid-training. During pre-training, we did not use best-fit packing [11] since random truncation is a good data-augmentation strategy for pre-training documents.For datasets in mid-training, we applied best-fit packing to avoid truncating the reasoning process or repo-level code.

# 2.4Hyper-Parameters

We employed the Muon optimizer [21; 24] for all parameters except word embedding, bias, and weights for RMSNorm. For hyperparameters,we set the Newton-Schulz iteration steps N to 5, momentum μ to 0.95,and scaled Muon's update RMS to 0.2. We observed that the Muon optimizer can accelerate convergence and tolerate larger batch sizes. We used cosine decay schedule for learning rate, instead of warmup-stable-decay (WSD) schedule [17]. Our early experiments showed that models trained with the WSD schedule perform worse on general benchmarks (SimpleQA, MMLU), indicating underfittng in the stable stage. The learning rate went through a warm-up stage from O to 2.5e-4 and a decaying stage to 2.5e-5 until the end of mid-training. We used a batch size warmup strategy, where the batch size was gradually increased from 16M tokens to 64M tokens in the training of the first 5OoB tokens,and remained constant in the remaining of training. For regularization, we set the weight decay ratio to O.1 and did not use dropout. We set the maximum sequence length to 4,096 during pre-training,and extended it to 32,768 and 131,072 during the mid-training stage as shown in Figure 3.When extending the sequence length to 32K, we also adjusted RoPE's base frequency from 10,000 to 1,0oo,0o0 for beter long-context modeling ability. For loss-free balance routing, we set the bias update rate to O.001 for the first 15T tokens,and to O.O for the remaining tokens. We also applied auxiliary sequence-level balance loss with a O.Ooo1 weight to avoid extreme imbalance within any single sequence.The MTP loss weight 入 was set to O.3 for the first 15T tokens, and to O.1 for the remaining tokens.

# 3Post-Training: Expert Model Iteration

We divide the post-training process into two distinct stages. In stage 1 (Expert Training), we construct expert models specializing in three domains: Reasoning, Agent, and General chat. In stage 2(Unified Training), we employ self-distillation techniques to integrate multiple experts,ultimately delivering a comprehensive model capable of generating responses through both deliberative reasoning and direct response modes.

# 3.1Supervised Fine-Tuning

We perform Supervised Fine-Tuning (SFT) at the beginning of both Stage 1 (Expert Training) and Stage 2 (Unified Training). In the expert training stage,the primary role of SFT is to provide a cold start, empowering the model with basic chat, reasoning,and tool-use capabilities,which can then be further enhanced in subsequent expert RL training to achieve improved performance. In the unified training stage, the purpose of SFT is to distil the capabilities of different expert models into one hybrid reasoning generalist capable of handling different types of tasks.

Cold Start SFTDuring the cold-start phase, we utilize a small set of supervised fine-tuning (SFT) data with extended Chain-of-Thought (CoT) responses. This approach ensures that each expert model possesses adequate foundational ability prior to the reinforcement learning phase.

Overall SFTIn the Overall SFT stage, we collect millions of samples covering reasoning tasks (math,code,science, etc.),general chat (writing, translation, summarization, chit chat, etc.),agentic tasks (basic tool using, coding ability especially for authentic project development, etc.),and longcontext understanding tasks from the previously trained expert models,and train the base model with a maximum context length of 128K tokens.By distilling from the output of distinct experts, the model learns to apply the most effective long CoT reasoning for each task to arrive at accurate answers. Especially,recognizing that a prolonged thinking process is unnecessary for certain domains that demand quick responses (such as chit chat), we meticulously balanced training data containing full reasoning with data lacking explicit thought processes. This approach allows the model to operate in both the reflective and immediate response modes,thereby creating a hybrid reasoning model. Moreover, we find the following strategy helpful in preparing SFTdata to derive optimal performance.

Reducing Character Escaping in Function Call TemplatesAlthough function call parameters are predominantly represented in JSON format in contemporary implementations,a significant challenge emerges when these parameters contain code segments. In such cases, a substantial proportion of characters within the code require escaping, compelling the model to generate extensive escape characters, thereby increasing the learning burden for the model. While this issue poses minimal concer for models primarily designed for general chat, it represents a non-trivial challenge for agentic foundation models where function caling is a core capability. To mitigate this limitation, we propose a novel function call template that encapsulates function call keys and values within XML-like special token tags. This approach substantially reduces the necessity for character escaping in code segments, as the vast majority of code can be represented in its native form without escaping. Experimental results demonstrate that the proposed function call template does not compromise the performance of function cal execution while recucing escaping. The following example (Figure 4) illustrates the structure of our proposed function cal tmplate. Detailed code implementation can be found in our open-source repository.

<|system|> # Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags: <tools>

{"name": "get_veather", "description": "Get the weather of a city for a specific date.", "parameters": {" type": "object", "properties": {"city": {"type": "string", "description": "The city to get weather for, in Chinese."}， "date": {"type": "string", "description": "The date"in YYY-MM-DD format."}}， "required": ["city"]}   
</tnnls> For each function call， output the function name and arguments within the following XML format: <tool_call>{function-name}   
<arg_key>{arg-key-1}</arg_key>   
<arg_value>{arg-value-1}</arg_value>   
<arg_key>{arg-key-2}</arg_key>   
<arg_value>{arg-value-2}</arg_value>   
···   
</tool_call><|system|>   
You are a helpful assistant.<|user|>   
Today is June 26， 2024. Could you please check the weather in Beijing and Shanghai for tomorrow<l assistant|>   
<think>The user wants to check the weather of Beijing and Shanghai tomorrow.I need to call the get_weather function respectively to check Beijing and Shanghai.</think>   
I will call the get_weather function to check the veather in Beijing and Shanghai.   
<tool_call>get_weather   
<arg_key>city</arg_key>   
<arg_value>Beijing</arg_value>   
<arg_key>date</arg_key>   
<arg_value>2024-06-27</arg_value>   
</tool_call>   
<tool_call>get_weather   
<arg_key>city</arg_key>   
<arg_value>Shanghai</arg_value>   
<arg_key>date</arg_key>

![](images/d3e0f39ae50adcdb0bef7eb90db6f04d4e8b655b0ecf759a0a31d8e0f5f942d0.jpg)  
Figure 4: One example of function call template.

Rejection Sampling When sampling from expert models, we employ a comprehensive multistage filtering pipeline that includes: (1) removing repetitive, excessively short, or truncated samples,as well as those that fail to conform to valid reasoning formats; (2) conducting correctness verification for samples with objective answers; (3) utilizing reward models to filter responses to subjective questions; and (4) for tool-calling scenarios, ensuring adherence to proper tool invocation protocols and verification that trajectories reach the expected terminal states.

Prompt Selection and Response-Level Scaling Filtering challenging prompts and conducting response scaling on them prove to be effective.We experimented with removing the prompts in the bottom 5O% based on response lengths,resulting in a 2%-4% improvement in math and science tasks,despite training with only half the data. Notably, we found that applying response scaling to these hard prompts can lead to further gains. Generating four responses for each prompt brought an additional 1%-2% improvement.

Automatic Agentic SFT Data Construction The construction of agentic SFT data involves four steps: 1. Agentic Framework and Tool Collection: We gather a set of agentic frameworks and real-world tool APIs and MCP servers, while also leveraging LLMs to automatically construct and simulate a batch of tools.2. Task Synthesis: Based on these frameworks and tools, we automatically synthesize a collection of agentic tasks. On the one hand,for relatively mature frameworks, we leverage LLMs to comprehend their functionalities and automatically generate relevant queries or tasks. On the other hand,for more fragmented or disparate tools,we first select a representative subset and similarly employ LLMs to construct tasks about this subset. These tasks encompas both single-step and multi-step toolcaling scenarios.3. Trajectory Generation: For each synthesized task, we utilize existing LLMs to generate tool-cal trajectories. Additionally,by employing the LLM as a user simulator, multi-step tool-cal tasks are converted into trajectories involving multiple rounds of dialogue. 4. Quality Filtering: For each trajectory, multiple judge agents are used to evaluate whether the task is completed. Only successful trajectories are retained.

# 3.2Reasoning RL

Reasoning RL focuses on enhancing a model's capabilities in domains that demand logical deduction, structured problem-solving,and verifiable accuracy. This includes critical areas such as mathematics, code generation,and scientific reasoning. A defining characteristic of these tasks is the high precision of their reward signals,as correctness can often be determined programmatically or with objective clarity. Mastery in these areas is not only crucial for advancing the raw intelligence of models but also serves as a fundamental building block for more complex, multi-step agentic behaviors. Recognizing the unique challenges and opportunities within reasoning RL, we have developed a suite of specialized techniques to effectively train our models. These methods,detailed below,are designed to address issues such as training efficiency, sample diversity,and data quality. Our overall RL algorithm builds upon the GRPO [31] framework, excluding the KL loss term. The comparison curves shown in this section are based on our smaller experimental model, not on GLM-4.5.

Difficulty-based Curriculum LearningDuring reinforcement learning, the model's proficiency evolves,creating a mismatch with static training data. In the later stages,as the model becomes more capable,overly simple data can lead to rollouts where all rewards are 1s. Conversely, in the early stages, excessively difficult data often results in batches where all rewards are Os. In both scenarios, the lack of reward variance provides no useful gradient signal, severely hindering training efficiency. To address this challenge, we employ a two-stage difficulty-based curriculum for RL.The effectiveness of this and other strategies discussed below is validated through controlled experiments on a smaller model, which allows for rapid iteration and precise ablation studies.As shown in Figure 5, this two-stage approach enables the model to consistently surpass its performance ceiling. Crucially, to maintain high signal quality and reduce noise,allproblems used in the second stage are strictly sourced from a pool with verified correct answers.

![](images/4ee4ff542ff6ae013bc5aafcc18e9d8e20dc8d47580b5f1eeef31046dd25e712.jpg)

![](images/61e78d3b8ce97355b0c4a82368ea5b3ffc38e8cfb6b238da2babe01bfa625f80.jpg)  
Figure 5: Effectiveness of the two-stage diffculty-based curriculum on AIME'24. The blue line (our method) switches to extremely difficult problems (pass @8=0, pass @512>O) in the second stage, showing continued improvement. The red line (baseline) continues with moderate-difficulty problems and plateaus.   
Figure 6: Single-stage vs. multi-stage RL at 64K context length. The red line (single-stage at 64K) achieves superior performance. The blue line (multi-stage with progressively increasing length) suffers from an irreversible performance drop in early stages,limiting its final performance.

Single-Stage RL at 64K Output Length Previous research [25] has suggested conducting RL in multiple stages with progressively increasing maximum output lengths. However, our experiments reveal that this multi-stage approach is less effective than a single-stage RL process conducted directly at the maximum target length of 64K. Since the initial Supervised Fine-Tuning (SFT) has already conditioned the model on generating 64K-length responses, introducing RL stages with shorter maximum lengths can cause the model to“unlearn” its long-context capabilities. This often leads to a significant and irreversible drop in performance,as the model's average output length decreases. This degradation is difficult to recover from in the final 64K-length RL stage, thus limiting further improvement. Our experiments confirm this observation: as demonstrated in Figure 6,applying RL directly at the full 64K-length continually pushes the model's limits and yields beter performance.

![](images/8bc915172eb2625fa7d7f06ff02b2a9ccd29501ada01f8b277508ef2105d57c1.jpg)  
Figure 7: Ablation Studies for Code and Science RL.(Left) Comparison of loss calculation methods for code RL. The token-weighted mean loss approach achieves faster convergence compared to the sequence-mean loss baseline, accelerating the training process. (Right) Ablation on data sources for science RL on the GPQA-Diamond benchmark. Training exclusively on a small set of high-quality, expert-verified multiple-choice questions yields the best performance,significantly outperforming training on mixed-quality data.

Dynamic Sampling TemperatureDuring RL, the sampling temperature is a key parameter for controlling trajectory diversity. A temperature that is too low leads to convergent, less exploratory outputs,while one that is too high introduces low-quality, noisy samples,undermining both model accuracy and training eficiency. Using a fixed sampling temperature is suboptimal because it fails to adapt as the policy distribution becomes more concentrated (i.e., has lower entropy),often resulting in insufficient exploration at later stages. Therefore, we propose dynamically adjusting the sampling temperature to maintain a healthy balance between accuracy and exploration. Specifically, when the average reward of rolouts stabilizes,we identify this as a convergence phase and increase the sampling temperature to encourage greater diversity. To mitigate the risk of introducing excessive noise, we implement a quality-control mechanism: we periodically evaluate model performance on a held-out validation set across a range of temperatures.The temperature for the next training phase is then set to the maximum value that does not cause a performance drop of more than 1% from the current optimum [2].

Code and Science RL Compared to mathematics, RL for coding and scientific domains has received less attention in the literature. We conducted extensive controlled RL experiments in these areas and arrived at the following empirical conclusions. For code RL,we find that the choice of loss calculation is critical for training efciency. As illustrated in Figure 7 (left), adopting a token-weighted mean loss is highly beneficial compared to a conventional sequence-mean loss. The token-weighted approach provides a finer-grained and more stable gradient signal, which leads to significantly faster convergence. This method also helps to alleviate the length bias inherent in sequence-level rewards and effectively suppresses the generation of overly simplistic or repetitive “base case” samples during training. For science RL,our findings on the GPQA-Diamond benchmark highlight that data quality and type are paramount factors. As shown in Figure 7 (right), using exclusively expert-verified multiple-choice questions for RL leads to significantly better performance compared to training with mixed-quality or unverified data. This result underscores that even for tasks with simple formats like multiple-choice,rigorously filtering the RL data pool to include only high-quality, challenging instances is crucial for effective model improvement.

# 3.3Agentic RL

Reinforcement Learning from Human Feedback (RLHF) helps language models follow human instructions more faithfully. Applying RL to math and programming contests has further uncovered strong reasoning abilities and favorable scaling behavior on tasks whose outcomes can be objectively verified. Building on these insights, we focus on agentic setings—specifically web-search and code-generation agents—where every action or answer can be automatically checked. This built-in verifiability supplies dense,reliable rewards, enabling us to scale RL training more effectively.

# 3.3.1Data Collection and Synthesis for Agents

For web-search tasks and open-domain information seeking, we develop a data-synthesis pipeline that yields demanding question-answer pairs requiring multi-step reasoning across multiple web sources. This corpus is designed to sharpen GLM's ability to uncover elusive,interwoven facts on the internet. Dataset construction blends two approaches: (1) an automated pipeline powered by multi-hop reasoning over knowledge graphs, and (2) human-in-the-loop extraction and selective obfuscation of content from several web pages to prepare reinforcement-learning training signals.

For software-engineering tasks,we curate an extensive collection of GitHub pull requests and issues to create a realistic software-development benchmark comprising user prompts and executable unit tests.All evaluations run inside a hardened sandbox with a distributed system, which provides both horizontal scalability and strong isolation guarantees.

# 3.3.2Pushing the Limits with Reinforcement Learning and Iterative Self-distillation

We adopt the group-wise policy optimization algorithm for RL training. For each problem x, we sample K agent traces {yi,.. , yk} from the previous policy Told, and optimize the model πe with respect to the following objective:

![](images/31df846f77da3287ae8008aaf64a4a31cee1f1ba2e970323e9e8e81cd88a98e6.jpg)

computation.

Outcome Supervision with Process Action Format PenaltyFor web search tasks, we use the accuracy of the final answer as a reward for the entire agent trace. For coding agents, we primarily utilize SWE data with verifiable test cases for RL training. Our experiments have shown that RL training on web search and SWE tasks leads to generalized performance improvements across other tasks and benchmarks,such as general tool usage and coding tasks like Terminal-Bench. Additionally, we apply a process format penalty to ensure the model generates correct tool call formats. If the model fails to produce the correct tool format during agent trace generation, the process will be halted,and the trace will receive a zero reward.

Iterative Distillation Since RL training on agent tasks is time-consuming, we adopt a selfdistillation approach to iteratively enhance the performance of the SFT cold-start model before resuming RL training on this improved model. Specifically, we first perform RL training on the initial cold-start model to boost agent performance. Once training has reached a certain step count or plateaued, we apply self-distilation by substituting the original cold-start data with responses generated by the RL-trained model, thus creating a superior SFT model. We then conduct further RL training on this enhanced model, progressively increasing training difficulty. This iterative strategy allows us to push the performance limits of RL-trained models efficiently.

Scaling Test-time Compute via Interaction TurnsFor agent tasks, we observe significant performance gains given increasing interaction turns with the environment. Compared to test-time scaling in reasoning models, which scales output tokens, agent tasks make use of test-time compute by continuously interacting with the environment, e.g., searching high and low for hard-to-find web information or writing test cases for self-verification and self-correction for coding tasks. Figure 8 shows that with varying browsing effort, accuracy scales smoothly with test-time compute.

# 3.4General RL

General RL aims to holistically improve the model's overal performance, remediate potential issues, and strengthen key capabilities. Central to our methodology is a multi-source feedback system that synergizes rule-based feedback, human feedback (RLHF),and model-based feedback (RLAIF). This hybrid framework provides more robust training signals and alows us to leverage the unique advantages of each source: the precision of automated rules, the nuanced judgment of human annotators,and the scalability of AI-driven evaluation.

![](images/2daf72ae7d95e0fa0304ce0d15eb344ac62a196500e7b6fb9f2028bf16aecb2a.jpg)  
Figure 8: Interaction Turns Scaling for BrowseComp.

![](images/77d4c171db5e37e3507e0afe9a804c9a9197a8849f81beb3e291c0b846eb5090.jpg)  
Figure 9: Training curve of Instruction Following RL without other General RL tasks. During GRPO training, the instruction following performance (SysBench-ISR) improves in step with the increasing reward. Up to roughly 1,0OO training steps, we have not observed clear evidence of reward hacking.

Holistic RLHolistic RL targets broad performance gains across diverse domains.To this end, we first construct a balanced dataset of roughly 5,0oO prompts spanning 7 primary,33 secondary,and 139 tertiary categories. Reward signals for Holistic RL are derived from both human and AI feedback. For human feedback, we train a reward model on preference annotations. Annotators compare model responses and assign preference labels based on a comprehensive evaluation of multiple dimensions, such as instruction following,safety,and factual correctness. For model feedback, we design separate scoring rubrics that depend on whether the prompt has an objective ground-truth answer. Merging the two feedback sources yields more reliable and expressive reward signals, mitigating the inherent limitations of each individual method.

Instruction Following RLInstruction Following RL improves the model's ability to understand and satisfy complex instructions.To achieve this, we create a fine-grained taxonomy with 7 major and 151 minor constraint types, covering content requirements, formatting rules,and more. Based on this taxonomy, a dedicated training set of challenging instructions is assembled to cover every constraint type. The feedback system consists of deterministic verification rules,a trained reward model, and a critique model. The robustness of this hybrid feedback system proves crucial during GRPO training. We observe mitigated reward hacking, enabling the policy model to achieve continuous and steady improvements in instruction following as shown in Figure 9.

Function Calling RLFunction Calling RL is divided into step-wise rule-based RL and end-to-end multi-turn RL. We incorporate step-wise rule-based RL directly into our general RL framework due to their similar output lengths and convergence speeds.For end-to-end multi-turn RL, we first train specialized expert models and then distill these experts into the main model.

· Step-wise Rule-based RL: For tasks with clear tool invocation procedures, we annotate the ground truth function callfor each step/turn in the training data. Given the task and the function calls from previous steps/turns,the model is trained to generate the next assistant response, which can be a function call or a response to the user. Using rule-based rewards, we guide the model to make correct function calls over consecutive rounds. Accordingly, we design the following strict reward function:

Reward = (1, if FormatCorrect(at) and Match(at,a\*) {0，otherwise

Here,at denotes the t-th function cal generated by the model, and a\* is the corresponding ground truth function call. A reward of 1 is only given if at is in the correct format and matches the ground truth exactly (including the name, parameters,and every field). Otherwise,the reward is O. Such a strict reward rule not only guides the model to generate correct function cals but also strongly enforces output formatting, improving the model's usability and robustness in real-world interactions.

·End-to-end Multi-turn RL: Step-wise rule-based RL decomposes tasks into static, predetermined decision flows.In this process, the model lacks dynamic interactions with the environment and cannot autonomously explore, plan,or handle complex situations, thereby making its real-world problem-solving ability limited. To address these issues,we introduce end-to-end multi-turn function calling RL, where the model first generates the complete trajectory and is then rewarded based on task completion. In this way, the model can optimize its action policy through continuous trial and error with tool feedback, significantly enhancing its ability in autonomous planning and decision-making. Specifically, end-to-end multi-turn function calling RL considers two types of complex tasks: 1. single-turn multi-step tasks: The model needs to make multi-step function calls and interact with the environment to complete such tasks. We use complex tasks automatically synthesized based on MCP servers, as well as some open-source agentic datasets with runnable environments,such as Agentgym [46]. 2. multi-turn multi-step tasks: Besides interacting with the tool execution environment, the model also needs to interact with an LLM-simulated user agent to obtain complete task information and accomplish the overall tsk. The reward for end-to-end multi-turn function calling RL is computed as:

Reward {1i

Here,I refers to the original complex task, at is the t-th function calland Ot is the tool feedback or user information. TaskCompleted(I,oo,a1,O1,...,aT,OT) indicates whether the task is completed, which is determined by the environment according to predefined rules or by an LLM Judge Agent.

Pathology RLAs the final stage of post-training, general RL needs to rectify potential issues, such as language mixing, excessive repetition, and formatting mistakes. Although penalizing such behaviors in the above-mentioned general RL tasks is effective,the low incidence rate of these pathologies (often less than 1% of outputs) makes this a sample-ineficient optimization strategy. Therefore, we curate a targeted dataset for pathology RL by identifying prompts that are highly likely to trigger these pathological behaviors. Training on this dataset lets us impose eficient penalties, further lowering the residual error rates for these problematic behaviors.

# 3.5RL Infrastructure

Our RL infrastructure is built upon Slime1,an open-source framework we developed. The framework is engineered with several key optimizations to enhance flexibility, efficiency,and extensibility.

Flexible Hybrid Training and Data Generation ArchitectureA core feature of our infrastructure is its support for highly flexible training paradigms and data generation strategies within a single, unified system. This design allows us to cater to the distinct requirements of various RL tasks by supporting both a colocated, synchronous mode and a disaggregated, asynchronous mode. This flexibility in data generation is crucial for extending our RL capabilities to new domains and more complex agentic environments. We observed that different RL tasks benefit from diffrent scheduling approaches. For general-purpose RL tasks or those aimed at enhancing model reasoning capabilities (e.g., in mathematics and code generation),a synchronous,colocated architecture is more effective. In this setup,the training and inference engines reside on the same worker. This,combined with dynamic sampling,significantly reduces GPU idle time and maximizes resource utilization. Conversely, for agentic tasks, such as those in Software Engineering (SWE), the data generation process is often protracted and involves complex system interactions. To ensure that the agent environments can operate continuously and maximize data throughput, we adopt a disaggregated,asynchronous model. The rollout component of the RL framework is exposed directly to the agent environment, while the GPUs for training and inference are scheduled independently. This decoupling enables the agent environments to constantly generate new data without being stalled by the training cycle. By leveraging the resource scheduling and asynchronous capabilities of the Ray framework, we can flexibly place the inference and training engines on the same GPU or on different ones. This dual support for synchronous and asynchronous training allows diverse RL tasks to share a common set of underlying optimizations for both training and inference.

![](images/c96360f44cf6a9fc7bb96d87bbbcd842ae427582ca7992cab95a900becfda601.jpg)  
Figure 10: Overview of the Slime RL infrastructure. The system consists of three core modules: Training (Megatron) - handles the main training process, reads data from the Data Buffer, and synchronizes parameters with the rollout module after training; Rollout (SGLang + Router) - generates new data, including rewards and verifier outputs,and writes it to the Data Buffer; Data Buffer- serves as a bridge module that manages prompt initialization, custom data,and rollout generation strategies.

Accelerated Rollut with Mixed-Precision Inference Rollout efficiency is a persistent bottleneck in RL training. To address this,our infrastructure supports BF16 for training while leveraging FP8 for inference to accelerate the data generation phase. During each policy update iteration, we perform online,block-wise FP8 quantization on the model parameters before they are dispatched for rollout. This dynamic quantization enables highly efficient FP8 inference,significantly improving the overall throughput of the data collection process.

Agent-oriented RL Infra Design To conduct RL for agent tasks, we design a fully asynchronous and decoupled RL infrastructure that efficiently handles long-horizon agent rollouts and supports flexible multi-task RL training across diverse agent frameworks.

Agentic rollouts often require prolonged interactions with complex environments, which can significantly slow down the overallRL training process. To overcome this, we first design a high-concurrency Docker-based runtime that provisions isolated environments for each task, drastically reducing rollout overhead. In addition, we implement a fully asynchronous RL training loop. Because agent tasks can vary in type and trajectory length,synchronous RL training often leads to severe GPU underutilization as workers wait for the slowest rollouts to complete. Our approach partitions GPUs into dedicated rollout engines and training engines: the rollout engines continuously generate trajectories, while the training engines update the model weights and periodically synchronize them back to the rollout engines. This decoupled design prevents long or diverse trajectories from blocking the entire training pipeline,resulting in consistently high throughput, particularly in scenarios with highly variable agent interactions.

Another key challenge is the diversity of existing agent frameworks, which are tailored to different tasks.Leveraging these frameworks not only improves task-specific performance but also maintains alignment between training and inference. To achieve this, we introduce a unified HTTP endpoint interface coupled with a centralized data pool. Since most agent frameworks produce rollouts in a message-list format,all trajectories are stored in this data pool, which serves as a shared source for training. This architecture cleanly decouples task-specific rollout logic from the RL training process, enabling seamless integration of heterogeneous agent frameworks. Furthermore,the data pool supports customizable,task-specific filtering and dynamic sampling strategies to ensure high-quality RL training data across diverse tasks.

Through these two core designs, our system provides a scalable, flexible,and high-performance solution for long-agentic RL,and can support long-horizon rollouts and adapt to a wide range of agent tasks.

# 4Evaluation

# 4.1Evaluation of Base Models

We first evaluate the performance of our base model GLM-4.5-Base. Table 2 shows the comparison results of the last checkpoint of pre-training of our base model. Please note that the base model has not been trained on instruction data, and the GLM-4.5-Base scores are from our internal evaluation framework. Results show that GLM-4.5-Base is stable on all the different benchmarks, including English, Code,Math,and Chinese,which validates our idea of unifying all the abilities into one model.

Table 2: Comparison among GLM-4.5-Base and other representative open-source base models.   

<table><tr><td></td><td>Benchmark (Metric)</td><td>Qwen3-235B -A22B Base</td><td>Llama4-Maverick 400B Base</td><td>DeepSeek-V3 Base</td><td>Kimi-K2 Base</td><td>GLM-4.5 Base</td></tr><tr><td rowspan="3"></td><td>Architecture</td><td>MoE</td><td>MoE</td><td>MoE</td><td>MoE</td><td>MoE</td></tr><tr><td># Activated Params</td><td>22B</td><td>17B</td><td>37B</td><td>32B</td><td>32B</td></tr><tr><td># Total Params</td><td>235B</td><td>400B</td><td>671B</td><td>1043B</td><td>355B</td></tr><tr><td rowspan="6">English</td><td>SimpleQA (EM)</td><td>-</td><td>-</td><td>26.6</td><td>35.3</td><td>30.0</td></tr><tr><td>BBH (EM)</td><td>88.9</td><td>87.1</td><td>88.4</td><td>88.7</td><td>86.2</td></tr><tr><td>MMLU (EM)</td><td>87.8</td><td>85.2</td><td>87.2</td><td>87.8</td><td>86.1</td></tr><tr><td>HellaSwag (EM)</td><td>-</td><td>-</td><td>88.9</td><td>94.6</td><td>87.1</td></tr><tr><td>PIQA (EM)</td><td></td><td>=</td><td>84.7</td><td>-</td><td>85.3</td></tr><tr><td>TriviaQA (EM)</td><td>-</td><td>-</td><td>82.9</td><td>85.1</td><td>80.0</td></tr><tr><td rowspan="2">Code</td><td>EvalPlus (Pass @1)</td><td>77.6</td><td>65.5</td><td>65.6</td><td>80.3</td><td>78.1</td></tr><tr><td>LiveCodeBench-Base (Pass @1)</td><td>-</td><td>25.1</td><td>24.6</td><td>26.3</td><td>28.1</td></tr><tr><td rowspan="2">Math</td><td>GSM8K (EM)</td><td>94.4</td><td>87.7</td><td>87.6</td><td>92.1</td><td>79.4</td></tr><tr><td>MATH (EM)</td><td>71.8</td><td>63.3</td><td>62.6</td><td>70.2</td><td>61.0</td></tr><tr><td rowspan="4">Chinese</td><td>CLUEWSC (EM)</td><td></td><td></td><td>82.7</td><td>-</td><td>83.5</td></tr><tr><td>C-Eval (EM)</td><td></td><td>80.9</td><td>90.1</td><td>92.5</td><td>86.9</td></tr><tr><td>C3 (EM)</td><td></td><td>-</td><td>78.6</td><td>-</td><td>83.1</td></tr><tr><td>Chinese-SimpleQA (EM)</td><td></td><td>53.5</td><td>72.1</td><td>77.6</td><td>70.1</td></tr></table>

# 4.2Evaluation on 12 (ARC) Benchmarks

We further evaluate our full GLM-4.5 models after Post-Training for allthe Agentic, Reasoning, and Coding (ARC) tasks,on 12 benchmarks: MMLU-Pro, AIME 24, MATH-500, SciCode, GPQA,HLE, LCB (2407-2501),SWE-Bench Verified, Terminal-Bench, TAU-Bench, BFCL V3, BrowseComp.

# 4.2.1Evaluation of Agentic Abilities

We evaluate the agentic abilities of GLM-4.5 in two aspects: TAU-bench [48] (including retail and airline domains)and Berkeley Function Call Leaderboard V3 (BFCL V3) [26], which measures the model's ability to call user-defined functions to respond to users’ queries. BrowseComp [45] measures the model's ability as a web browsing agent to find correct answers for complicated questions. For

Table 3: Results on Agentic Benchmarks. TAU represents TAU-bench [48] and BFCL represents Berkeley Function Calling Leaderboard [26].   

<table><tr><td>Benchmark</td><td>GLM- 4.5</td><td>GLM- 4.5-Air</td><td>03</td><td>04 mini</td><td>GPT-4.1</td><td>Claude Opus 4</td><td>Claude Sonnet 4</td><td>Gemini 2.5 Pro</td><td>Kimi K2</td><td>Grok 4</td></tr><tr><td>TAU-Retail</td><td>79.7</td><td>77.9</td><td>70.4</td><td>65.6</td><td>75.1</td><td>81.4</td><td>80.5</td><td>77.0</td><td>73.9</td><td>76.5</td></tr><tr><td>TAU-Airline</td><td>60.4</td><td>60.8</td><td>52.0</td><td>49.2</td><td>48.8</td><td>59.6</td><td>60.0</td><td>48.0</td><td>51.2</td><td>58.4</td></tr><tr><td>BFCL V3</td><td>77.8</td><td>76.4</td><td>72.4</td><td>67.2</td><td>68.9</td><td>74.4</td><td>75.2</td><td>61.2</td><td>71.1</td><td>66.2</td></tr><tr><td>BrowseComp</td><td>26.4</td><td>21.3</td><td>49.7</td><td>28.3</td><td>4.1</td><td>18.8</td><td>14.7</td><td>7.6</td><td>7.9</td><td>32.6</td></tr><tr><td>Average</td><td>58.1</td><td>55.7</td><td>61.1</td><td>50.1</td><td>45.0</td><td>54.6</td><td>53.4</td><td>43.8</td><td>47.2</td><td>55.4</td></tr></table>

TAU-bench, we use an optimized user simulator (Cf. Figure 11) for both the Retail and Airline domains.The user prompt we use can be found below in Figure 11. On TAU-bench, GLM-4.5's performance is better than Gemini 2.5 Pro and close to Claude Sonnet 4. On BFCL V3, GLM-4.5 achieves the best overall score among the baselines. On BrowseComp,the performance of OpenAI o3 is much better than that of other models. GLM-4.5's performance is close to the second-best model (o4-mini) and significantly better than Claude Opus 4.

You are a user interacting with an agent.{instruction_display}

- Just generate one line at a time to simulate the user's message.   
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.   
- Do not hallucinate information that is not provided in the instruction. Follow these guidelines: 1. If the agent asks for information NOT in the instruction:   
- Say you don't remember or don't have it   
Offer alternative information that IS mentioned in the instruction   
2. Examples:   
- If asked for order ID (not in instruction): ('Sorry, I don't remember the order ID, can you search for it? My name/email/phone number/zipcode is ...',   
- If asked for email (not in instruction): (I don't have my email handy, but I can give you my name and zip code which are...",   
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.   
Try to make the conversation as natural as possible，and stick to the   
personalities in the instruction.

# Constraint Handling:

Provide requests strictly based on what is explicitly stated in the instruction. Do not assume，extend, substitute,or generalize in any form.   
Do not modify or relax constraints on:   
Time / Date   
Budget   
Specific terms (e.g.，‘'same'’ must not be replaced with ‘'similar'')   
Core Rule: Any attribute NOT mentioned in the instruction can be either changed

or kept the same - Examples:

- If instruction says ‘exchange red item to blue'': Only color must change, other attributes (size，material，etc.）are flexible   
- If instruction says ‘'exchange red item to blue， keep the same size'': Both color must change AND size must stay the same   
- Exception: Only follow additional constraints when explicitly stated in the instruction   
# When NOT to finish the conversation:   
- Do not end until you have clearly and completely expressed all your requirements and constraints.   
- Do not end until the agent has completed all tasks mentioned in the instruction and verified no operations were missed.   
- Do not end if the agent's execution results do not match your expectations or are incorrect/incomplete.   
# When you CAN finish the conversation: - Only when all above conditions are satisfied AND all tasks are completed correctly.   
- OR when you have clearly expressed complete requirements but the system   
explicitly states it cannot complete them due to technical limitations - in this case，accept transfer to human.   
# How to finish the conversation:   
- If the agent has completed all tasks, generate ‘‘###ST0P###'’ as a standalone message without anything else to end the conversation.   
# Note:   
- You should carefully check if the agent has completed all tasks mentioned in the instruction before generating ‘‘###SToP###''.

# 4.2.2Evaluation of Reasoning

We evaluate the reasoning abilities of GLM-4.5 and GLM-4.5-Air on seven benchmarks, including MMLU-Pro [43], AIME 24,MATH 500 [14], SciCode [36], GPQA [30], Humanity's Last Exam (HLE) [28],and LiveCodeBench (LCB)[19]2.For the AIME and GPQA benchmarks, we report the average accuracy over 32 and 8 samples, respectively (Avg@32, Avg@8), to mitigate result variance. An LLM was used for automated answer validation. For the HLE benchmark,only the text-based questions were evaluated, with correctness judged by GPT-4o. Our evaluation code is also opensourced3. We also compute the average reasoning performance on the seven benchmarks with the intelligence index proposed by Artificial Analysis4. GLM-4.5 outperforms OpenAI o3 on AIME 24 and SciCode. On average, GLM-4.5 outperforms Claude Opus 4 and is close to DeepSeek-R1-0528.

Table 4: Results on Reasoning Benchmarks. HLE represents Humanity's Last Exam [28] and LCB represents LiveCodeBench (2407-2501) [19].   

<table><tr><td>Benchmark</td><td>GLM- 4.5</td><td>GLM- 4.5-Air</td><td>03</td><td>Claude Opus 4</td><td>Gemini 2.5 Pro</td><td>DeepSeek R1 0528</td><td>Qwen3 235B 2507</td><td>Grok 4</td></tr><tr><td>MMLU Pro</td><td>84.6</td><td>81.4</td><td>85.3</td><td>87.3</td><td>86.2</td><td>84.9</td><td>84.5</td><td>86.6</td></tr><tr><td>AIME 24</td><td>91.0</td><td>89.4</td><td>90.3</td><td>75.7</td><td>88.7</td><td>89.3</td><td>94.1</td><td>94.3</td></tr><tr><td>MATH 500</td><td>98.2</td><td>98.1</td><td>99.2</td><td>98.2</td><td>96.7</td><td>98.3</td><td>98.0</td><td>99.0</td></tr><tr><td>SciCode</td><td>41.7</td><td>37.3</td><td>41.0</td><td>39.8</td><td>42.8</td><td>40.3</td><td>42.9</td><td>45.7</td></tr><tr><td>GPQA</td><td>79.1</td><td>75.0</td><td>82.7</td><td>79.6</td><td>84.4</td><td>81.3</td><td>81.1</td><td>87.7</td></tr><tr><td>HLE</td><td>14.4</td><td>10.6</td><td>20.0</td><td>11.7</td><td>21.1</td><td>14.9</td><td>15.8</td><td>23.9</td></tr><tr><td>LCB</td><td>72.9</td><td>70.7</td><td>78.4</td><td>63.6</td><td>80.1</td><td>77.0</td><td>78.2</td><td>81.9</td></tr><tr><td>AA-Index (Est.)</td><td>67.7</td><td>64.8</td><td>70.0</td><td>64.4</td><td>70.5</td><td>68.3</td><td>69.4</td><td>73.2</td></tr></table>

# 4.2.3Evaluation of Coding

Table 5:Results on SWE-bench Verified and Terminal-Bench   

<table><tr><td>Benchmark</td><td>GLM- 4.5</td><td>GLM- 4.5-Air</td><td>03</td><td>GPT-4.1</td><td>Claude Opus 4</td><td>Claude Sonnet 4</td><td>Gemini 2.5 Pr0</td><td>DeepSeek R1 0528</td><td>Kimi K2</td></tr><tr><td>SWE-bench Verified Terminal-Bench</td><td>64.2</td><td>57.6</td><td>69.1</td><td>48.6</td><td>67.8</td><td>70.4</td><td>49.0</td><td>41.4</td><td>65.4</td></tr><tr><td></td><td>37.5</td><td>30.0</td><td>30.2</td><td>30.3</td><td>43.2</td><td>35.5</td><td>25.3</td><td>17.5</td><td>25.0</td></tr><tr><td>Average</td><td>50.9</td><td>43.8</td><td>49.7</td><td>39.5</td><td>55.5</td><td>53.0</td><td>37.2</td><td>29.5</td><td>45.2</td></tr></table>

To measure GLM-4.5's ability to complete real-world coding tasks, we evaluate it on two challnging benchmarks,SWE-bench Verified [20] and Terminal-Bench [35].SWE-bench measures the model's ability to modify an existing codebase to solve a GitHub issue.The Verified subset is a humanfiltered subset of 5O0 instances. For evaluation, we use OpenHands [42] v0.34.O with runs limited to

100 iterations and history truncation to prevent exceeding the 128K context limit, configured with temperature=O.6,top_p=1.0. Terminal-Bench measures the model's ability to accomplish complex tasks in a terminal environment. We use the Terminus framework and standard function calling rather than direct prompting for evaluation. On SWE-bench Verified, GLM-4.5 outperforms GPT-4.1 and Gemini-2.5-Pro. On Terminal-Bench, GLM-4.5 outperforms Claude Sonnet 4. On average, GLM-4.5 is the best competitor for Claude Sonnet 4 on coding tasks.

# 4.2.4Evaluation of General Abilities

Table 6: Results on Commonly Used General Chat Benchmarks   

<table><tr><td>Benchmark</td><td>GLM- 4.5</td><td>GLM- 4.5-Air</td><td>GPT-4.1</td><td>Claude Sonnet 4</td><td>Gemini 2.5 Pro</td><td>Grok 4</td><td>Qwen3 235B</td><td>Deepseek R1 0528</td><td>DeepSeek V3 0324</td><td>Kimi K2</td></tr><tr><td>MMLU</td><td>90.0</td><td>87.4</td><td>90.2</td><td>91.9</td><td>91.9</td><td>91.9</td><td>90.2</td><td>89.9</td><td>89.1</td><td>89.5</td></tr><tr><td>SimpleQA</td><td>26.4</td><td>14.5</td><td>42.3</td><td>18.5</td><td>54.0</td><td>51.9</td><td>45.8</td><td>27.8</td><td>27.7</td><td>31.0</td></tr><tr><td>IFEval</td><td>86.1</td><td>86.3</td><td>87.4</td><td>88.7</td><td>90.8</td><td>92.4</td><td>87.8</td><td>80.0</td><td>83.4</td><td>89.8</td></tr><tr><td>SysBench</td><td>81.0</td><td>77.4</td><td>80.6</td><td>80.6</td><td>82.2</td><td>81.5</td><td>83.3</td><td>81.2</td><td>79.8</td><td>79.0</td></tr><tr><td>MultiChallenge</td><td>52.8</td><td>42.5</td><td>38.3</td><td>55.3</td><td>57.5</td><td>65.2</td><td>58.2</td><td>46.5</td><td>37.0</td><td>54.1</td></tr></table>

To evaluate the model's general abilities, we employed a set of widely-adopted open-source benchmark datasets, encompassing knowledge-intensive evaluations MMLU (EM)[14] and SimpleQA (Correct)[44],and instruction-following assessments IFEval (Prompt Strict) [52],SysBench (ISR) [29], and MultiChallenge [1O]. MultiChallenge is a multi-turn conversational benchmark evaluating LLMs across four integrated capability dimensions. SysBench systematically evaluates LLMs’ system message following capabilities across multi-turn conversations through three-level granularity metrics. On the MMLU benchmark, nearly all flagship models, including GLM-4.5,demonstrate performance at a comparable level. SimpleQA,which reflects the factual knowledge of a model, shows that GLM-4.5 (355B) performs similarly to DeepSeek V3 and R1 (both 671B), despite having nearly half of the parameters. On the IFEval benchmark, GLM-4.5 outperforms DeepSeek R1. In the Sysbench evaluation, GLM-4.5 surpasses GPT-4.1, DeepSeek V3,and Kimi K2. Additionally, on the MultiChallenge benchmark, it demonstrates superior performance compared to both GPT-4.1 and DeepSeek R1.

# 4.2.5Evaluation of Safety

To systematically assess the safety alignment of our model, we utilized SafetyBench [51], a comprehensive benchmark designed to evaluate the safety of large language models. SafetyBench consists of 11,435 multiple-choice questions covering seven distinct categories of safety concerns, with data in both English and Chinese.This benchmark enables a standardized and scalable evaluation of a model's ability to handle potentially harmful or sensitive topics.The categories include Ethics and Morality, Illegal Activities,Mental Health,Offensiveness,Physical Health,Privacy and Property,and Unfairness and Bias.We evaluated GLM-4.5 against a suite of other leading models.The results indicate that GLM-4.5 achieves a strong safety score, competitive with other top-tier models. Its overall score of 89.87 is comparable to that of Kimi-K2 (90.48) and GPT-4.1 (89.71). Notably, GLM-4.5 demonstrates robust performance in the areas of Ethics and Morality (94.33), Mental Health (94.67),and Physical Health (96.67). While it performs well in preventing responses related to Illegal Activities (90.97) and protecting Privacy and Property (92.00), there is still room for improvement in addressing Unfairness and Bias,an area of ongoing focus for our development efforts.The detailed performance breakdown is presented in the table below.

# 4.3Evaluations for Hands-on Experience

Sometimes,a trained LLM may overfit some predefined benchmarks, which makes the evaluated results not precisely reflect real-world experience. To overcome this challenge and to gauge our model's performance in more realistic situations,we have established a comprehensive manual evaluation framework. Human evaluation is particularly advantageous for assessing performance on open-ended questions, where aspects like coherence, relevance, and creativity are paramount. This hands-on approach allows for a more granular analysis,enabling us to better pinpoint areas of weakness and understand the qualitative aspects of model behavior that automated metrics often miss.

Table 7: Evaluation Results on SafetyBench   

<table><tr><td>Model</td><td>Average</td><td>Ethics &amp; Morality</td><td>Illegal Activities</td><td>Mental Health</td><td>Offensiveness</td><td>Health</td><td>Property</td><td>Physical Privacy &amp; Unfairness &amp;Bias</td></tr><tr><td>GLM-4.5</td><td>89.9</td><td>94.3</td><td>91.0</td><td>94.7</td><td>83.0</td><td>96.7</td><td>92.0</td><td>77.4</td></tr><tr><td>GLM-4.5-Air</td><td>87.8</td><td>91.0</td><td>90.3</td><td>92.7</td><td>83.3</td><td>92.3</td><td>90.3</td><td>74.7</td></tr><tr><td>Gemini 2.5 Pro</td><td>90.5</td><td>94.7</td><td>91.7</td><td>95.3</td><td>84.3</td><td>97.0</td><td>92.3</td><td>78.0</td></tr><tr><td>Kimi K2</td><td>90.5</td><td>93.0</td><td>93.3</td><td>95.0</td><td>90.3</td><td>97.3</td><td>93.0</td><td>71.3</td></tr><tr><td>GPT-4.1</td><td>89.7</td><td>92.0</td><td>94.3</td><td>95.3</td><td>85.3</td><td>95.7</td><td>91.3</td><td>74.0</td></tr><tr><td>DeepSeek-V3-0324</td><td>88.8</td><td>92.3</td><td>90.7</td><td>95.0</td><td>84.7</td><td>95.7</td><td>91.0</td><td>72.3</td></tr><tr><td>DeepSeek-R1-0528</td><td>83.5</td><td>87.0</td><td>81.7</td><td>86.7</td><td>77.7</td><td>92.0</td><td>85.7</td><td>73.7</td></tr></table>

# 4.3.1Evaluation of General Chat

To test the practical application capabilities of our models, we curated a diverse dataset of realscenario user prompts. These prompts span multiple languages and cover a wide range of categories, including Mathematics, Text Processing, Text Generation, Subjective QA, Objective QA,Logical Reasoning,and Code Instructions.We meticulously filtered this collection to ensure high quality and appropriate difficulty, while also removing any data that could compromise user privacy or safety. The final dataset consists of 66O prompts, with a distribution of 392 in English,108 in Chinese,and 160 in other languages. For prompts requiring factual knowledge, we annotated the correct answers to serve as a ground truth for evaluation.

We conducted a comparative evaluation between GLM-4.5, Deepseek-R1-0528, and Kimi K2. For each prompt, the responses from the different models were presented in a randomized order to eliminate any potential sequential bias.A single, consistent evaluator then scored each response on a scale of O to 1O. This method of using the same evaluator at the same time for a batch of comparisons is designed to minimize deviations arising from diferent individual preferences and subjective standards.The reasoning contents of GLM-4.5 and Deepseek-R1-0528 are not presented to the evaluators. The average scores for each model across the diferent categories and languages are presented below.

English ResultsIn the English prompt set, GLM-4.5 achieved the highest overallscore. It demonstrated particularly strong performance in Mathematics, Objective QA,and Text Generation.   
Table 8: Human Evaluation Scores on English Prompts. Subj. stands for Subjective. Obj stands for Objective.Text Gen. stands for Text Generation.   

<table><tr><td>Model</td><td>Overall</td><td>Math</td><td>Text Proc.</td><td>Subj. QA</td><td>Obj. QA</td><td>Text Gen.</td><td>Logic</td><td>Code</td></tr><tr><td>GLM-4.5</td><td>8.66</td><td>8.72</td><td>8.00</td><td>8.36</td><td>8.82</td><td>8.61</td><td>9.25</td><td>8.53</td></tr><tr><td>DeepSeek-R1-0528</td><td>8.62</td><td>8.56</td><td>8.27</td><td>7.91</td><td>9.00</td><td>7.83</td><td>9.07</td><td>8.65</td></tr><tr><td>Kimi-K2</td><td>8.13</td><td>7.22</td><td>8.00</td><td>7.45</td><td>8.86</td><td>7.06</td><td>7.07</td><td>8.71</td></tr></table>

Chinese ResultsFor the Chinese prompts, GLM-4.5 again led with the highest average score, showing standout performance in Text Generation, Logical Reasoning,and Code Instructions.

Table 9: Manual Evaluation Scores on Chinese Prompts   

<table><tr><td>Model</td><td>Overall</td><td>Math</td><td>Text Proc.</td><td>Subj. QA</td><td>Obj. QA</td><td>Text Gen.</td><td>Logic</td><td>Code</td></tr><tr><td>GLM-4.5</td><td>8.37</td><td>7.68</td><td>8.20</td><td>8.50</td><td>8.66</td><td>9.00</td><td>9.27</td><td>8.89</td></tr><tr><td>DeepSeek-R1-0528</td><td>8.05</td><td>7.76</td><td>8.07</td><td>8.00</td><td>7.89</td><td>8.59</td><td>9.00</td><td>8.67</td></tr><tr><td>Kimi-K2</td><td>7.03</td><td>7.37</td><td>6.43</td><td>7.71</td><td>6.45</td><td>8.28</td><td>7.55</td><td>8.26</td></tr></table>

Other Languages ResultsIn the multilingual evaluation covering other languages, GLM-4.5 maintained its lead, excelling in Text Generation and Subjective QA.

Table 10: Manual Evaluation Scores on Other Language Prompts   

<table><tr><td>Model</td><td>Overall</td><td>Math</td><td>Text Proc.</td><td>Text Gen.</td><td>Subj. QA</td><td>Obj. QA</td><td>Code</td><td>Logic</td></tr><tr><td>GLM-4.5</td><td>8.49</td><td>8.67</td><td>8.13</td><td>8.90</td><td>9.33</td><td>8.71</td><td>7.86</td><td>8.33</td></tr><tr><td>DeepSeek-R1-0528</td><td>8.27</td><td>9.44</td><td>8.38</td><td>7.86</td><td>9.44</td><td>8.22</td><td>7.64</td><td>8.17</td></tr><tr><td>Kimi-K2</td><td>6.63</td><td>7.22</td><td>6.38</td><td>7.62</td><td>7.78</td><td>6.22</td><td>6.68</td><td>7.17</td></tr></table>

# 4.3.2Evaluation of Coding Agent

![](images/26e4e6f469bf861467380578d24de3a8184261eeb6fc9203d1f78fa9ce7fafd6.jpg)  
GLM-4.5's Experience with Agentic Coding in Real-world Development Scenarios

![](images/0eb25401416f6e24212ace37087eb6de605a5833502510a6152f1a8dfb1d52d2.jpg)  
Figure 12: Head-to-head evaluation results between GLM-4.5 and other models on CC-Bench.   
Figure 13: Average tool calling success rate and token usage per interaction across different models on CC-Bench.

Experimental SetupTo evaluate the agentic coding capabilities of GLM-4.5 in real-world scenarios, we constructed CC-Bench, a benchmark built on the Claude Code5,encompassing 52 carefully designed programming tasks across diverse software development domains6. We compare GLM-4.5 against three strong baselines: Claude Sonnet 4, Kimi K2,and Qwen3-Coder. Each task was executed in an isolated containerized environment to prevent cross-task interference, with models initialized using predefined API configurations. Testing was conducted interactively by human experts over multiple rounds: each task began with a standardized prompt, followed by iterative interactions where experts adjusted inputs based on model outputs until the task was completed or failed. To ensure fairness,the same expert followed consistent interaction strategies across all models.

Based on this testing procedure, model performance was evaluated using the following criteria: the primary metric was task completion, determined by predefined completion criteria.In cases of ties,efficiency and reliability—including tool calling success rate and token consumption efficiency—were used as secondary metrics. The evaluation prioritized functional correctness and task completion over eficiency metrics,ensuring that coding capability remained the primary evaluation focus.

ResultsIn head-to-head evaluations, GLM-4.5 demonstrated strong performance relative to opensource baselines and competitive capability against closed-source models as shown in figure 12. Specifically:

· GLM-4.5 vs Claude Sonnet 4: 40.4% win, 9.6% tie, 50.0% loss · GLM-4.5 vs Kimi K2: 53.9% win, 17.3% tie, 28.8% loss · GLM-4.5 vs Qwen3-Coder: 80.8% win, 7.7% tie, 11.5% loss

As shown in figure 13, GLM-4.5 particularly exceled in tool calling reliability,achieving the highest success rate at 90.6%, compared to Claude Sonnet 4 (89.5%), Kimi-K2 (86.2%),and Qwen3-Coder (77.1%). While Claude Sonnet 4 remains a strong competitor, GLM-4.5 outperformed other models in both task completion consistency and agentic execution robustness.

# 4.3.3Evaluation of Logical Reasoning

To rigorously assess the true logical reasoning capabilities of the models and mitigate the risk of data contamination from common logical questions found online, we constructed a new, chalenging evaluation set. This set comprises novel and complex logical reasoning problems that are structurally different from those widely available on the internet. Each problem is designed to require multiple steps of logical deduction to arrive at the correct solution.

For this evaluation, we established a unified and detailed scoring standard for each question. We then tasked each model with solving these problems. The correctness and quality of each model's response were subsequently inspected and scored by human experts. The results show a competitive landscape, with GLM-4.5 performing on par with leading models.

Table 11: Expert Evaluation Scores on Novel Logical Reasoning Problems   

<table><tr><td>Model</td><td>Score</td></tr><tr><td>Gemini 2.5 Pro</td><td>65.8</td></tr><tr><td>DeepSeek-R1-0528</td><td>62.1</td></tr><tr><td>GLM-4.5</td><td>62.0</td></tr><tr><td>GLM-4.5-Air</td><td>53.4</td></tr><tr><td>Kimi K2</td><td>51.9</td></tr></table>

# 4.4Evaluation of Translation

The New Paradigm of TranslationTranslation today extends beyond simple text conversion to encompass a nuanced understanding of evolving internet slang, cultural context, and domain-specific terminology:

Netizen Lingo: Translating“yyds” accurately requires recognizing it as the acronym for the Chinese phrase“永远的神”(yong yuan de shén), meaning“the eternal god,”thus capturing its true sentiment of enthusiastic praise and admiration.

Domain Nicknames: Recognizing“胖”(literally “fat white") is critical within photography communities. Specialized models may translate it incorrectly, but a general-purpose model understands it as a widely used nickname for the“Canon EF 70-300mm f/4-5.6 Is UsM" lens, providing precise translations.

Symbols: When a Chinese user sends a“fish” emoji in a conversation to refer to a second-hand marketplace,can the model understand the cultural meme behind it, which points to the“闲鱼” (Xianyu) platform? This tests the model's cognitive ability to connect visual symbols with online cultural phenomena.

Deep Contextual Reasoning:Translating“三花公主驾到，速来围观”demands identifying“三花” not as a person's name but as a reference to the popular calico coloration of cats. A general-purpose model accurately deduces this context,translating the phrase idiomatically as “The Calico Princess has arrived! Come and see!".

These examples underscore modern translation as a task rooted deeply in knowledge and reasoning.

Evaluation ResultsWe tested 1OO challenging, real-world cases commonly mistranslated by current tools, comparing GLM-4.5 against specialized translation models (Qwen-MT-plus, QwenMT-turbo, Seed-X [9]) in a blind human evaluation (scored O-3 considering whether the meaning is conveyed correctly and whether the language is authentic). The results are shown in Table 12.

Table 12: Human Scores on Selected Challenging Translation data   
GLM-4.5 significantly outperforms specialized models.For example,translating“三花公主驾到” specialized models failed contextually, whereas GLM-4.5 accurately conveys the idiomatic meaning.   

<table><tr><td>Model</td><td>Average Score</td></tr><tr><td>GLM-4.5</td><td>1.71</td></tr><tr><td>Qwen-MT-plus</td><td>0.38</td></tr><tr><td>Qwen-MT-turbo</td><td>0.55</td></tr><tr><td>Seed-X</td><td>0.65</td></tr></table>

# 5Conclusion

In this report, we have introduced the GLM-4.5 model series, including GLM-4.5 and GLM-4.5-Air. Both models adopt the MoE architecture, which improves the computational efciency compared to previous GLM models. GLM-4.5 excels at reasoning,coding,and agentic tasks,ranked in 3rd place globally among open-source and proprietary models. We release the model weights of GLM-4.5 and GLM-4.5-Air to advance the applications and research of large language models.

# 6Contribution

Contributors’ names are listed in alphabetical order by first name. Names marked with an asterisk (\*) indicate individuals who have since left our team.

# Core Contributors

Bin Chen, Chengxing Xie, Cunxiang Wang,Da Yin, Hao Zeng, Jiajie Zhang,Kedong Wang, Lucen Zhong,Mingdao Liu,Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, Yean Cheng, Yifan An, Yilin Niu, Yuanhao Wen,Yushi Bai, Zhengxiao Du, Zihan Wang (汪子涵),Zilin Zhu

# Contributors

Bohan Zhang, Bosi Wen, Bowen Wu,Bowen Xu\*, Can Huang, Casey Zhao, Changpeng Cai, Chao Yu, Chen Li, Chendi Ge,Chenghua Huang, Chenhui Zhang, Chenxi Xu, Chenzheng Zhu, Chuang Li\*, Congfeng Yin, Daoyan Lin, Dayong Yang, Dazhi Jiang, Ding Ai, Erle Zhu, Fei Wang, Gengzheng Pan, Guo Wang, Hailong Sun, Haitao Li, Haiyang Li, Haiyi Hu, Hanyu Zhang, Hao Peng, Hao Tai, Haoke Zhang, Haoran Wang, Haoyu Yang\*,He Liu, He Zhao, Hongwei Liu, Hongxi Yan, Huan Liu, Huilong Chen, JiLi, Jiajing Zhao, Jiamin Ren, Jian Jiao, Jiani Zhao, Jianyang Yan, Jiaqi Wang\*, Jiayi Gui, Jiayue Zhao, JieLiu,JijieLi, Jing Li, Jing Lu, Jingsen Wang, Jingwei Yuan,Jingxuan Li, Jingzhao Du,Jinhua Du,Jinxin Liu,Junkai Zhi, Junli Gao,Ke Wang,Lekang Yang\*,Liang Xu,Lin Fan,Lindong Wu, Lintao Ding,Lu Wang, Man Zhang, Minghao Li, Minghuan Xu, Mingming Zhao, Mingshu Zhai\*,Pengfan Du, Qian Dong, Shangde Lei, Shangqing Tu, Shangtong Yang,Shaoyou Lu, Shijie Li, Shuang Li (李泷), Shuang Li(李爽), Shuxun Yang, Sibo Yi\*, Tianshu Yu, Wei Tian, Weihan Wang, Wenbo Yu, Weng Lam Tam, Wenjie Liang, Wentao Liu, Xiao Wang\*, Xiaohan Jia, Xiaotao Gu, Xiaoying Ling, Xin Wang, Xing Fan, Xingru Pan, Xinyuan Zhang, Xinze Zhang, Xiuqing Fu, Xunkai Zhang, Yabo Xu, Yandong Wu, Yida Lu, Yidong Wang, Yilin Zhou, Yiming Pan, Ying Zhang, Yingli Wang, Yingru Li, Yinpei Su, Yipeng Geng, Yitong Zhu, Yongkun Yang\*, Yuhang Li, Yuhao Wu\*, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yuxuan Zhang, Zezhen Liu, Zhen Yang, Zhengda Zhou, Zhongpei Qiao, Zhuoer Feng, Zhuorui Liu, Zichen Zhang, Zihan Wang (王梓 汉), Zijun Yao, Zikang Wang, Ziqiang Liu, Ziwei Chai, Zixuan Li, Zuodong Zhao\*

Tech Leads Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou

# Advisors

Jie Tang, Yuxiao Dong, Juanzi Li, Hongning Wang,Minlie Huang,Bin Xu,Jidong Zhai, Wenguang Chen

# Acknowledgement

We are grateful for all the support from Beijing, Shanghai, Tianjin, Hangzhou, Zhuhai,and Chengdu.   
Special thanks to our customers and community developers.

# References

[1] A. Abbas, K. Tirumala, D. Simig, S. Ganguli, and A. S. Morcos. Semdedup: Data-eficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540, 2023.   
[2] C. An, Z. Xie, X. Li,L. Li,J. Zhang, S. Gong,M. Zhong,J. Xu, X. Qiu, M. Wang, and L. Kong. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025.   
[3] Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 3119-3137,2024.   
[4] Y. Bai, S. Tu, J. Zhang,H. Peng, X. Wang, X. Lv,S. Cao, J. Xu,L. Hou, Y. Dong, J. Tang, and J. Li. LongBench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3639-3664, Vienna, Austria, July 2025. Association for Computational Linguistics.   
[5] M. Bavarian,H. Jun, N. Tezak, J. Schulman, C.McLeavey, J. Tworek,and M. Chen. Eficient training of language models to fill in the middle, 2022.   
[6] T. Brown,B. Mann,N. Ryder,M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan,P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.   
[7] A. Chen, A. Li, B. Gong, B. Jiang, B. Fei, B. Yang, B. Shan, C. Yu, C. Wang, C. Zhu, et al. Minimax-m1: Scaling test-time compute eficiently with lightning attention. arXiv preprint arXiv:2506.13585,2025.   
[8] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374,2021.   
[9] S. Cheng, Y. Bao, Q. Cao, L. Huang, L. Kang, Z. Liu, Y. Lu, W. Zhu, Z. Huang, T. Li, et al. Seed-x: Building strong multilingual translation llm with 7b parameters. arXiv preprint arXiv:2507.13618,2025.   
[10] K. Deshpande, V. Sirdeshmukh,J. B. Mols,L. Jin,E.-Y. Hernandez-Cardona,D. Lee, J. Kritz, W. E. Primack, S. Yue, and C. Xing. Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier llms. In Findings of the Association for Computational Linguistics: ACL 2025, pages 18632-18702, 2025.   
[11] H. Ding, Z. Wang, G. Paolini, V. Kumar,A. Deoras,D. Roth,and S. Soatto. Fewer truncations improve language modeling. In Proceedings of the 41st International Conference on Machine Learning, pages 11030-11048, 2024.   
[12]F. Gloeckle, B. Y. Idrissi, B. Roziere,D. Lopez-Paz,and G. Synnaeve. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024.   
[13] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-rl: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[14] D. Hendrycks,C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song,and J. Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirty-fth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).   
[15] A. Henry,P.R. Dachapally, S.Pawar,and Y. Chen. Query-key normalization for transformers, 2020.   
[16] C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya,D. Rekesh,F. Jia,and B. Ginsburg. Ruler: What’s the real context size of your long-context language models? In First Conference on Language Modeling.   
[17] S. Hu, Y. Tu, X. Han, G. Cui, C. He, W. Zhao, X. Long, Z. Zheng, Y. Fang, Y. Huang, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. In First Conference on Language Modeling.   
[18] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carmey, et al. Openai ol system card. arXiv preprint arXiv:2412.16720, 2024.   
[19] N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations.   
[20] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press,and K. Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.   
[21] K.Jordan，Y. Jin，V. Boza，Y. Jiacheng，F. Cecista，L.Newhouse，and J.Bernstein..Muon:An optimizer for hidden layers in neural networks,2024. URL https://kellerjordan.github.io/posts/muon, 6.   
[22] A. Joulin,E. Grave,P.Bojanowski,and T.Mikolov. Bag of tricks for effcient text classification. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, pages 427-431. Association for Computational Linguistics, April 2017.   
[23] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.   
[24] J. Liu, J. Su, X. Yao, Z. Jiang, G. Lai, Y. Du, Y. Qin, W. Xu, E. Lu, J. Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982,2025.   
[25] M. Luo, S. Tan, J. Wong, X. Shi, W. Y. Tang, M. Roongta, C. Cai, J. Luo, L. E. Li, R. A. Popa, and I. Stoica.Deepscaler: Surpassng o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Modelby-Scaling-RL-19681902c1468005bed8ca303013a4e2,2025. Notion Blog.   
[26] S. G.Patil, H. Mao, C. Cheng-Jie Ji,F. Yan, V. Suresh,I. Stoica, and J. E. Gonzalez. The berkeley functioncalling leaderboard (bfcl):From tooluse to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025.   
[27] G. Penedo, H. Kydlicek, V. Sabolcec,B. Messmer, N. Foroutan, A. H. Kargaran, C. Raffel, M.Jaggi,L.VonWerrandT.Wolf.Fineweb2:Onepipelinetoscalethemalldapting pre-training data processing to every language. arXiv preprint arXiv:2506.20920, 2025.   
[28] L.Phan, A. Gati, Z. Han, N. Li, J. Hu, H. Zhang, C. B. C. Zhang, M. Shaaban, J. Ling, S. Shi, et al. Humanity's last exam. arXiv preprint arXiv:2501.14249, 2025.   
[29] Y. Qin, T. Zhang, Y. Shen, W. Luo, Y. Zhang, Y. Qiao, Z. Zhou, W. Zhang, B. CUI, et al. Sysbench: Can llms follow system message? In The Thirteenth International Conference on Learning Representations,2024.   
[30] D. Rein, B.L. Hou,A. C. Stickland,J. Pety, R. Y. Pang, J. Dirani,J. Michael, and S. R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.   
[31] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300,2024.   
[32] D. Su, K. Kong, Y. Lin,J. Jennings,B. Norick,M. Kliegl, M. Patwary, M. Shoeybi, and B.Catanzaro. Nemotron-c: Transforming commoncrawlintoarefined long-horizon pretraining dataset. arXiv preprint arXiv:2412.02595, 2024.   
[33] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805,2023.   
[34] K. Team, Y. Bai, Y. Bao, G. Chen, J. Chen, N. Chen, R. Chen, Y.Chen, Y. Chen, Y. Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.   
[35] T.T.-B.Team. Terminal-bench: A benchmark for ai agents in terminal environments,Apr 2025.   
[36] M. Tian,L. Gao,S. Zhang, X. Chen, C. Fan, X. Guo, R. Haas, P. Ji, K. Krongchon, Y.Li, et al. Scicode: A research coding benchmark curated by scientists. Advances in Neural Information Processing Systems,37:30624-30650, 2024.   
[37] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T.Lacroix,B. Roziere, N. Goyal, E. Hambro,F. Azhar, et al. Llama: Open and effcient foundation language models. arXiv preprint arXiv:2302.13971,2023.   
[38] K. Vodrahali, S. Ontanon, N. Tripuraneni, K. Xu, S. Jain, R. Shivanna,J. Hui, N. Dikkala, M. Kazemi, B. Fatemi, R. Anil, E. Dyer, S. Shakeri, R. Vij, H. Mehta, V. Ramasesh, Q. Le, E. Chi,Y.Lu, O.Firat, A. Lazaridou, J.-B.Lespiau,N. Attaluri,and K. Olszewska. Michelangelo: Long context evaluations beyond haystacks via latent structure queries, 2024.   
[39] F. Wan, W. Shen, S. Liao, Y. Shi, C.Li, Z. Yang,J. Zhang,F. Huang, J. Zhou, and M. Yan. Qwenlong-l1: Towards long-context large reasoning models with reinforcement learning. arXiv preprint arXiv:2505.17667, 2025.   
[40] L. Wang, H. Gao, C. Zhao, X. Sun, and D. Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts. arXiv preprint arXiv:2408.15664, 2024.   
[41] S. Wang, L. Yu, C. Gao, C. Zheng, S. Liu, R. Lu, K. Dang, X. Chen, J. Yang, Z. Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025.   
[42] X. Wang, B.Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, H. H. Tran, F.Li, R. Ma, M. Zheng, B. Qian, Y. Shao,N. Muennighoff, Y. Zhang, B. Hui, J. Lin, R. Brennan, H. Peng, H. Ji, and G. Neubig. Openhands: An open platform for AI software developers as generalist agents. In The Thirteenth International Conference on Learning Representations, 2025.   
[43] Y. Wang, X. Ma, G. Zhang, Y. Ni,A. Chandra,S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266-95290, 2024.   
[44] J. Wei, N. Karina, H. W. Chung, Y. J. Jiao, S. Papay, A. Glaese, J. Schulman, and W. Fedus. Measuring short-form factuality in large language models, 2024.   
[45] J. Wei, Z. Sun, S. Papay,S. McKinney, J. Han,I. Fulford, H. W. Chung,A. T. Passos, W. Fedus, and A. Glaese. Browsecomp: A simple yet chalenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516,2025.   
[46] Z. Xi, Y. Ding, W. Chen, B. Hong, H. Guo, J. Wang, D. Yang, C. Liao, X. Guo, W. He, et al. Agentgym: Evolving large language model-based agents across diverse environments. arXiv preprint arXiv:2406.04151,2024.   
[47] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C.Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.   
[48] S. Yao, N. Shinn, P. Razavi, and K. Narasimhan. tau-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045,2024.   
[49] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai,T. Fan, G. Liu,L. Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.   
[50] A. Zeng, X. Liu, Z. Du, Z. Wang, H. Lai, M. Ding, Z. Yang, Y. Xu, W. Zheng, X. Xia, et al. Glm-130b: An open bilingual pre-trained model. In The Eleventh International Conference on Learning Representations.   
[51] Z. Zhang, L. Lei,L. Wu, R. Sun, Y.Huang, C. Long, X. Liu, X. Lei, J. Tang, and M. Huang. Safetybench: Evaluating the safety of large language models with multiple choice questions. arXiv preprint arXiv:2309.07045,2023.   
[52] J. Zhou,T.Lu, S. Mishra, S.Brahma, S. Basu, Y.Luan,D. Zhou,and L. Hou. Instructionfollowing evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.