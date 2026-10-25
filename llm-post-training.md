# LLM Post-Training with Imperfect Information

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/> <img src="https://img.shields.io/badge/Number%20of%20Items-374-FF6F00" alt="PaperNum"/>

A curated list of recent papers (2024 – now) on **post-training large language models with imperfect information**.

**Scope.** A paper is included when the information used to post-train an LLM (supervised fine-tuning, RLHF, DPO-style preference optimization, RL with rewards or verifiers, distillation) is imperfect: preference labels are noisy or inconsistent, reward models or verifiers are wrong in places, ground-truth labels are weak or missing, fine-tuning or teacher data contain errors, or the feedback has been corrupted — and handling this imperfection is the paper's main contribution. Generic alignment methods that assume clean supervision, and multimodal models, are out of scope.

Papers are from top conferences and journals; a few recent, notable arXiv preprints are included and marked `arXiv`. Within each section, the newest papers come first.

This list is part of [Advances in Learning with Imperfect Information](./README.md), curated and maintained by [Dong-Dong Wu](https://wu-dd.github.io/). You can also [browse, search and filter it online](https://wu-dd.github.io/Advances-in-Learning-with-Imperfect-Information/#llm). Suggestions are welcome by email.

+ :octocat: Code (187 papers have official code)

## Latest Updates
+ [2026/10/25] First release.

<!-- papers:start -->
## Contents
- [Imperfect Preference Data](#imperfect-preference-data)
  - [Noisy Preference Labels](#noisy-preferences)
  - [Ambiguous & Inconsistent Preferences](#ambiguous-preferences)
  - [Noisy AI Feedback & LLM-as-a-Judge](#noisy-ai-feedback)
  - [Weak / Sparse Preference Supervision](#weak-preference-supervision)
- [Imperfect Rewards & Verifiers](#imperfect-rewards-verifiers)
  - [Reward Hacking & Over-Optimization](#reward-hacking)
  - [Noisy Verifiers & Rewards in RLVR](#noisy-verifiers)
  - [Imperfect Process Reward Models](#imperfect-prm)
  - [Reward Uncertainty & Misspecification](#reward-uncertainty)
- [Weak & Missing Supervision](#weak-missing-supervision)
  - [Weak-to-Strong Generalization](#weak-to-strong)
  - [Label-Free & Self-Rewarding RL](#label-free-rl)
  - [Partial, Negative & Incomplete Supervision](#partial-supervision)
  - [Learning from Pseudo-Labels](#pseudo-labels)
- [Imperfect SFT & Distillation Data](#imperfect-sft-distillation-data)
  - [Noisy Labels in Fine-Tuning](#noisy-sft)
  - [Flawed Demonstrations & Reasoning Traces](#flawed-reasoning-data)
  - [Imperfect Teachers in Distillation](#imperfect-teacher)
  - [Synthetic Data, Self-Improvement & Model Collapse](#synthetic-data)
  - [Imperfect Agent Trajectories](#imperfect-trajectories)
- [Corrupted Feedback & Theory](#corrupted-feedback-theory)
  - [Poisoned Preference & Reward Data](#poisoned-feedback)
  - [Corruption-Robust Post-Training](#corruption-robust)
  - [Theory](#theory)

<a name="imperfect-preference-data" />

## <u>Imperfect Preference Data</u>

<a name="noisy-preferences" />

### <u>Noisy Preference Labels</u>

- [Correct, Route, Calibrate: Efficient Preference Optimization from Noisy, Heterogeneous Human Feedback](https://openreview.net/forum?id=UOoH4aqCJg) (NeurIPS 2026)
- [Learning Where It Matters: Geometric Anchoring for Robust Preference Alignment](https://openreview.net/forum?id=Ockny0oU9m) (NeurIPS 2026)
- [Oracle-Robust Online Alignment for Large Language Models](https://openreview.net/forum?id=gd1SmOZK7y) (NeurIPS 2026)
- [PORT: Preference Optimization via Robust Token-Level Reweighting](https://openreview.net/forum?id=Z0gcomg5dI) (NeurIPS 2026)
- [Robust Nash Alignment under Preference Uncertainty](https://openreview.net/forum?id=CiLFISXIZl) (NeurIPS 2026)
- [PLC-DPO: Posterior Label Correction in Noisy and Ambiguous Preference Optimization](https://arxiv.org/abs/2608.30597) (EMNLP 2026) [:octocat:](https://github.com/VennTum99/PLC-DPO)
- [Optimal Transport for LLM Reward Modeling from Noisy Feedback](https://openreview.net/forum?id=InoePQ7HNI) (ICML 2026)
- [Reliability-Aware LLM Alignment from Inconsistent Human Feedback](https://openreview.net/forum?id=0LeyqHkrEG) (ICML 2026) [:octocat:](https://github.com/ExploreXploitQ/RGPO)
- [TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](https://openreview.net/forum?id=YDztzMJynP) (ICML 2026) [:octocat:](https://github.com/abdulhadyabas2/TUR-DPO-Topology--and-Uncertainty-Aware-Direct-Preference-Optimization)
- [Task-Aware Preference Calibration for Direct Preference Optimization](https://openreview.net/forum?id=GxWTX93etl) (ICML 2026)
- [Unbiased Alignment for Large Language Models with Noisy Preferences](https://openreview.net/forum?id=eMQsdioK8z) (ICML 2026) [:octocat:](https://github.com/cswjl/unbiased-alignment)
- [RE-PO: Robust Enhanced Policy Optimization as a General Framework for LLM Alignment](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b292ef48c211e2d8288146ff045dd61d-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/XiaoyangCao1113/RE-PO)
- [Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8836d0294651a867c2ca51012923997-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/SkyworkAI/Skywork-Reward-V2)
- [How Well Can Preference Optimization Generalize Under Noisy Feedback?](https://openreview.net/forum?id=8f5gRWwzDx) (TMLR 2026)
- [On Symmetric Losses for Policy Optimization with Noisy Preferences](https://openreview.net/forum?id=cBWGLmSeao) (TMLR 2026) [:octocat:](https://github.com/nissymori/SymPO)
- [RMO: Towards Better LLM Alignment via Reshaping Reward Margin Distributions](https://ojs.aaai.org/index.php/AAAI/article/view/40565) (AAAI 2026)
- [Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b457a310c34efcaf6322d1bc6503ce16-Abstract-Datasets_and_Benchmarks_Track.html) (NeurIPS 2025) [:octocat:](https://github.com/deeplearning-wisc/PrefCleanBench)
- [Less is More: Improving LLM Alignment via Preference Data Selection](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ebf95a6f3c575322da15d4fd0fc2b3c8-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/xiangtanshi/DPO-Data-Selection)
- [Scalable Valuation of Human Feedback through Provably Robust Model Alignment](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c2c4c24889d8d0e4b0b0ce4a98fd56ef-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/ma921/HolderDPO)
- [Adaptive Preference Optimization with Uncertainty-aware Utility Anchor](https://aclanthology.org/2025.findings-emnlp.1046/) (EMNLP 2025)
- [DORM: Preference Data Weights Optimization for Reward Modeling in LLM Alignment](https://aclanthology.org/2025.findings-emnlp.1237/) (EMNLP 2025)
- [Energy-Based Reward Models for Robust Language Model Alignment](https://openreview.net/forum?id=x6evCULIOQ) (COLM 2025) [:octocat:](https://github.com/AnamikaLochab/EBRM)
- [ROPO: Robust Preference Optimization for Large Language Models](https://proceedings.mlr.press/v267/liang25d.html) (ICML 2025)
- [Robust Preference Optimization via Dynamic Target Margins](https://aclanthology.org/2025.findings-acl.282/) (ACL 2025) [:octocat:](https://github.com/sunjie279/gammaPO)
- [Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment](https://arxiv.org/abs/2505.10597) (arXiv 2025)
- [Model Editing as a Robust and Denoised variant of DPO: A Case Study on Toxicity](https://proceedings.iclr.cc/paper_files/paper/2025/hash/acb7ce5aab6e134300a2361dd90a501f-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/Uppaal/detox-edit)
- [Towards Robust Alignment of Language Models: Distributionally Robustifying Direct Preference Optimization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/294fd30f8cbb89ce328d5a01fef47fb4-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/junkangwu/Dr_DPO)
- [Perplexity-aware Correction for Robust Alignment with Noisy Preferences](https://proceedings.neurips.cc/paper_files/paper/2024/hash/31a57804448363bcab777f818f75f5b4-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/luxinyayaya/PerpCorrect)
- [β-DPO: Direct Preference Optimization with Dynamic β](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea888178abdb6fc233226d12321d754f-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/junkangwu/beta-DPO)
- [Enhancing Language Model Alignment: A Confidence-Based Approach to Label Smoothing](https://aclanthology.org/2024.emnlp-main.1189/) (EMNLP 2024)
- [Filtered Direct Preference Optimization](https://aclanthology.org/2024.emnlp-main.1266/) (EMNLP 2024) [:octocat:](https://github.com/CyberAgentAILab/filtered-dpo)
- [Reward Modeling Requires Automatic Adjustment Based on Data Quality](https://aclanthology.org/2024.findings-emnlp.234/) (EMNLP 2024)
- [Impact of Preference Noise on the Alignment Performance of Generative Language Models](https://openreview.net/forum?id=nMAaCsCTCI) (COLM 2024)
- [Provably Robust DPO: Aligning Language Models with Noisy Feedback](https://proceedings.mlr.press/v235/ray-chowdhury24a.html) (ICML 2024)

<a name="ambiguous-preferences" />

### <u>Ambiguous & Inconsistent Preferences</u>

- [Distributionally Robust Listwise Preference Optimization](https://openreview.net/forum?id=FL7ACXQ5Rb) (NeurIPS 2026)
- [Response Time Enhances Alignment with Heterogeneous Preferences](https://openreview.net/forum?id=lSacVQkH6p) (NeurIPS 2026)
- [Act-Adaptive Margin: Dynamically Calibrating Reward Models for Subjective Ambiguity](https://aclanthology.org/2026.acl-long.710/) (ACL 2026) [:octocat:](https://github.com/calubkk/AAM)
- [The Sign Estimator: Preference Modeling for LLM Alignment under Heterogeneity](https://openreview.net/forum?id=SyTeWTEex3) (ICML 2026)
- [When Labelers Stay Silent: The Power of Ties in Cost-Effective Preference Learning](https://openreview.net/forum?id=dXY80mjA7D) (ICML 2026)
- [Keep the Best, Forget the Rest: Reliable Alignment with Order-Aware Preference Optimization](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bd5dd89a1a2f49ec1bdbbf5c1b5be9c4-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/pxyWaterMoon/rappo)
- [On Extending Direct Preference Optimization to Accommodate Ties](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4dbf3488a8da908a8177d525ede1a43a-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/EriChen0615/DPO-RKD)
- [ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/17e51d497e8bacaa79582e98cd043f1a-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/timokau/response-rank)
- [Diverging Preferences: When do Annotators Disagree and do Models Know?](https://proceedings.mlr.press/v267/zhang25bx.html) (ICML 2025)
- [Reward Modeling with Ordinal Feedback: Wisdom of the Crowd](https://proceedings.mlr.press/v267/liu25az.html) (ICML 2025)
- [MallowsPO: Fine-Tune Your LLM with Preference Dispersions](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4ddab70bf41ffe5d423840644d3357f4-Abstract-Conference.html) (ICLR 2025)
- [Adaptive Preference Scaling for Reinforcement Learning with Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c1f66abb52467443ba8fc70e0a32e061-Abstract-Conference.html) (NeurIPS 2024)
- [Geometric-Averaged Preference Optimization for Soft Preference Labels](https://proceedings.neurips.cc/paper_files/paper/2024/hash/688c7a82e31653e7c256c6c29fd3b438-Abstract-Conference.html) (NeurIPS 2024)
- [Margin Matching Preference Optimization: Enhanced Model Alignment with Granular Feedback](https://aclanthology.org/2024.findings-emnlp.792/) (EMNLP 2024) [:octocat:](https://github.com/kykim0/margin-matching-pref-opt)
- [Distributional Preference Learning: Understanding and Accounting for Hidden Context in RLHF](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7423902b5534e2b267438c85444a54b1-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/cassidylaidlaw/hidden-context)

<a name="noisy-ai-feedback" />

### <u>Noisy AI Feedback & LLM-as-a-Judge</u>

- [CoAct: Co-Active LLM Preference Learning with Human-AI Synergy](https://aclanthology.org/2026.acl-long.751/) (ACL 2026)
- [Curriculum-RLAIF: Curriculum Alignment with Reinforcement Learning from AI Feedback](https://aclanthology.org/2026.findings-acl.1685/) (ACL 2026) [:octocat:](https://github.com/ljy2222/Curriculum-RLAIF)
- [When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d1fa901403156f5d31b1cad964d78006-Abstract-Conference.html) (ICLR 2026)
- [Stackelberg Self-Annotation: A Robust Approach to Data-Efficient LLM Alignment](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5af741d487c5f0b08bfe56e11d1883e4-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/EunTilofy/SSAPO)
- [Optimising Factual Consistency in Summarisation via Preference Learning from Multiple Imperfect Metrics](https://aclanthology.org/2025.findings-emnlp.940/) (EMNLP 2025) [:octocat:](https://github.com/Haruhi07/MultiMetric)
- [Voting with the Graph: Stable RLAIF via Topological Consistency Maximization](https://arxiv.org/abs/2510.15514) (arXiv 2025)
- [Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback](https://aclanthology.org/2025.acl-long.355/) (ACL 2025) [:octocat:](https://github.com/allenai/hybrid-preferences)
- [RLTHF: Targeted Human Feedback for LLM Alignment](https://proceedings.mlr.press/v267/xu25c.html) (ICML 2025)
- [Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment](https://proceedings.iclr.cc/paper_files/paper/2025/hash/342e5fc02b86dec9b24e41b22968e539-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/kingdy2002/SPA)
- [Your Weak LLM is Secretly a Strong Teacher for Alignment](https://proceedings.iclr.cc/paper_files/paper/2025/hash/332b4fbe322e11a71fa39d91c664d8fa-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/deeplearning-wisc/weak_llm_teacher)
- [Self-Evolutionary Large Language Models Through Uncertainty-Enhanced Preference Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/34724) (AAAI 2025) [:octocat:](https://github.com/wjn1996/Uncertainty-Preference-Optimization)

<a name="weak-preference-supervision" />

### <u>Weak / Sparse Preference Supervision</u>

- [From “Weak” Signals to Strong Models: Preference Delta Aggregation with LoRA Merging](https://openreview.net/forum?id=NTLZwZkDzg) (NeurIPS 2026) [:octocat:](https://github.com/AlbertQiSun/Preference-Delta-Aggregation)
- [Ultra-DPO: Reimagining LLM Alignment as a Semi-Supervised Task](https://openreview.net/forum?id=11tDIVzbAR) (NeurIPS 2026)
- [Personalizing LLMs with Binary Feedback: A Preference-Calibrated Optimization Framework](https://aclanthology.org/2026.acl-long.1222/) (ACL 2026)
- [Unbiased Reward Modeling from Implicit Feedback for LLM Alignment](https://openreview.net/forum?id=membbcuXeR) (ICML 2026)
- [WildReward: Learning Reward Models from In-the-Wild Human Interactions](https://aclanthology.org/2026.acl-long.864/) (ACL 2026) [:octocat:](https://github.com/THU-KEG/WildReward)
- [DRIFT: Learning from Abundant User Dissatisfaction in Real-World Preference Learning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c40bed606c51c8e827c1ba75aa2da054-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/cacayaya/DRIFT)
- [Semi-Supervised Preference Optimization with Limited Feedback](https://proceedings.iclr.cc/paper_files/paper/2026/hash/543ec10715d964122ab7cb15f648772b-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/MLAI-Yonsei/SSPO)
- [Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f2184e55a13b73b89f618ad24abb6ca7-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/deeplearning-wisc/LENS)
- [User Feedback in Human-LLM Dialogues: A Lens to Understand Users But Noisy as a Learning Signal](https://aclanthology.org/2025.emnlp-main.133/) (EMNLP 2025) [:octocat:](https://github.com/lyh6560new/implicit-user-feedback)
- [The Delta Learning Hypothesis: Preference Tuning on Weak Data can Yield Strong Gains](https://openreview.net/forum?id=9rwtezthwo) (COLM 2025) [:octocat:](https://github.com/scottgeng00/delta_learning)
- [Semi-Supervised Reward Modeling via Iterative Self-Training](https://aclanthology.org/2024.findings-emnlp.434/) (EMNLP 2024) [:octocat:](https://github.com/RLHFlow/RLHF-Reward-Modeling)
- [Model Alignment as Prospect Theoretic Optimization](https://proceedings.mlr.press/v235/ethayarajh24a.html) (ICML 2024) [:octocat:](https://github.com/ContextualAI/HALOs)

<a name="imperfect-rewards-verifiers" />

## <u>Imperfect Rewards & Verifiers</u>

<a name="reward-hacking" />

### <u>Reward Hacking & Over-Optimization</u>

- [Designing Effective Monitor-Based Interventions for Mitigating Reward Hacking During RL](https://openreview.net/forum?id=NERzW49Ahm) (NeurIPS 2026)
- [Mitigating Reward Hacking via Task Representations](https://openreview.net/forum?id=KYN2kU8Mva) (NeurIPS 2026)
- [Reward Bias Substitution: Single-Axis Bias Mitigations Redirect Optimization Pressure](https://openreview.net/forum?id=Auv9cgHOSa) (NeurIPS 2026) [:octocat:](https://github.com/maxlampe/bias_substitution)
- [Reward Hacking in Rubric-Based Reinforcement Learning](https://openreview.net/forum?id=Gor618iz3H) (NeurIPS 2026)
- [Automatically Finding Reward Model Biases](https://openreview.net/forum?id=Xy4ClJMjIU) (ICML 2026) [:octocat:](https://github.com/chry-santhemum/reward-model-bias)
- [Bias Fitting to Mitigate Length Bias of Reward Model in RLHF](https://aclanthology.org/2026.acl-long.133/) (ACL 2026)
- [Debiasing Reward Models via Causally Motivated Inference-Time Intervention](https://aclanthology.org/2026.acl-long.1029/) (ACL 2026)
- [Factored Causal Representation Learning for Robust Reward Modeling in RLHF](https://openreview.net/forum?id=CwNItJ07ew) (ICML 2026) [:octocat:](https://github.com/CMACH508/CausalRM)
- [Gradient Regularization Mitigates Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards](https://openreview.net/forum?id=T67db38qhr) (ICML 2026) [:octocat:](https://github.com/JohannesAck/gradientregularization_trl)
- [How RLHF Amplifies Sycophancy](https://openreview.net/forum?id=XN4pWKtA5h) (ICML 2026)
- [One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models](https://openreview.net/forum?id=20c6jemABm) (ICML 2026) [:octocat:](https://github.com/drfein/OneBiasAfterAnother)
- [Teach a Reward Model to Correct Itself: Reward Guided Adversarial Failure Discovery for Robust Reward Modeling](https://aclanthology.org/2026.acl-long.418/) (ACL 2026) [:octocat:](https://github.com/pankayaraj/ACL_2026_REFORM)
- [Thinking-Based Non-Thinking: Solving the Reward Hacking Problem in Training Hybrid Reasoning Models via Reinforcement Learning](https://aclanthology.org/2026.acl-long.2122/) (ACL 2026)
- [Unbiased Principles, Robust Rewards](https://openreview.net/forum?id=VZDlkXuIQc) (ICML 2026) [:octocat:](https://github.com/ShadeCloak/IP-GRM)
- [Disentangling Length Bias in Preference Learning via Response-Conditioned Modeling](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9713d53ee4f31781304b1ca43266f8d1-Abstract-Conference.html) (ICLR 2026)
- [Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d2f6f1dfbf9cd89a78c5a58ef0dec245-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/Qwen-Applications/DIR)
- [Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2096bafd3073a2224f6f0adb594068df-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/anirudhb123/Preference-Model-Biases)
- [ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ca688eb14e29701a11bdba6633186328-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/safety-research/impossiblebench)
- [Is it Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d0ee47d732bd526c4a03d4aecc39bc1-Abstract-Conference.html) (ICLR 2026)
- [Robust Reward Modeling via Causal Rubrics](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fa0a1013e171c1be6121c9bc9fb6589f-Abstract-Conference.html) (ICLR 2026)
- [Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d4b892df8bec8e13925f57649fbae039-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/PedroMLF/Reward-Hacking-in-Explanations)
- [Mitigating Reward Over-optimization in Direct Alignment Algorithms with Importance Sampling](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8f6c674a7edf57912f1a5830f4a66e3f-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/duyhominhnguyen/IS-DAAs)
- [Rectifying Shortcut Behaviors in Preference-based Reward Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5d97b7e62022c859347397f6c1e8d0f9-Abstract-Conference.html) (NeurIPS 2025)
- [Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397) (arXiv 2025)
- [reWordBench: Benchmarking and Improving the Robustness of Reward Models with Transformed Inputs](https://aclanthology.org/2025.emnlp-main.167/) (EMNLP 2025)
- [Adversarial Training of Reward Models](https://openreview.net/forum?id=H6Ae8Po6fS) (COLM 2025)
- [Off-Policy Corrected Reward Modeling for Reinforcement Learning from Human Feedback](https://openreview.net/forum?id=0zxugBcgF5) (COLM 2025) [:octocat:](https://github.com/JohannesAck/OffPolicyCorrectedRewardModeling)
- [MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking](https://proceedings.mlr.press/v267/farquhar25a.html) (ICML 2025) [:octocat:](https://github.com/google-deepmind/mona)
- [On the Robustness of Reward Models for Language Model Alignment](https://proceedings.mlr.press/v267/hong25d.html) (ICML 2025) [:octocat:](https://github.com/LinkedIn-XFACT/RM-Robustness)
- [Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization](https://aclanthology.org/2025.acl-long.649/) (ACL 2025) [:octocat:](https://github.com/kimsh0507/rethinking_rm_eval)
- [The Energy Loss Phenomenon in RLHF: A New Perspective on Mitigating Reward Hacking](https://proceedings.mlr.press/v267/miao25c.html) (ICML 2025) [:octocat:](https://github.com/miaoyuchun/Energy-Loss-Phenomenon)
- [Beyond Excess and Deficiency: Adaptive Length Bias Mitigation in Reward Models for RLHF](https://aclanthology.org/2025.findings-naacl.169/) (NAACL Findings 2025)
- [Language Models Learn to Mislead Humans via RLHF](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b9a5a60573637f329b04d1beda4cd404-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/Jiaxin-Wen/MisleadLM)
- [Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1646e34971facbcda3727d1dc28ab635-Abstract-Conference.html) (ICLR 2025)
- [Post-hoc Reward Calibration: A Case Study on Length Bias](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5d50c76fdf75c24ece568fc84a7125fb-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/ZeroYuHuang/Reward-Calibration)
- [RRM: Robust Reward Model Training Mitigates Reward Hacking](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9d46574e5baae5121180228223a11836-Abstract-Conference.html) (ICLR 2025)
- [Sail into the Headwind: Alignment via Robust Rewards and Dynamic Labels against Reward Hacking](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c78f81a878a72566422f37279bca0fd0-Abstract-Conference.html) (ICLR 2025)
- [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926) (arXiv 2025)
- [Reward Shaping to Mitigate Reward Hacking in RLHF](https://arxiv.org/abs/2502.18770) (arXiv 2025) [:octocat:](https://github.com/PorUna-byte/PAR)
- [InfoRM: Mitigating Reward Hacking in RLHF via Information-Theoretic Reward Modeling](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f25d75fc760aec0a6174f9f5d9da59b8-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/miaoyuchun/InfoRM)
- [Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/71f7154547c748c8041505521ca433ab-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/YangRui2015/Generalizable-Reward-Model)
- [Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e45caa3d5273d105b8d045e748636957-Abstract-Conference.html) (NeurIPS 2024)
- [A Long Way to Go: Investigating Length Correlations in RLHF](https://openreview.net/forum?id=G8LaO1P0xv) (COLM 2024) [:octocat:](https://github.com/prasanns/rlhf-length-biases)
- [Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking](https://openreview.net/forum?id=5u1GpUkKtG) (COLM 2024) [:octocat:](https://github.com/google-deepmind/reward-ensembles)
- [Countering Reward Over-Optimization in LLM with Demonstration-Guided Reinforcement Learning](https://aclanthology.org/2024.findings-acl.740/) (ACL Findings 2024) [:octocat:](https://github.com/MathieuRita/llm_demonstration_guided_rl)
- [Iterative Data Smoothing: Mitigating Reward Overfitting and Overoptimization in RLHF](https://proceedings.mlr.press/v235/zhu24e.html) (ICML 2024)
- [ODIN: Disentangled Reward Mitigates Hacking in RLHF](https://proceedings.mlr.press/v235/chen24bn.html) (ICML 2024)
- [WARM: On the Benefits of Weight Averaged Reward Models](https://proceedings.mlr.press/v235/rame24a.html) (ICML 2024)
- [Confronting Reward Model Overoptimization with Constrained RLHF](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5eee634cb9729b8bcc2ec9f2a46a74ae-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/tedmoskovitz/ConstrainedRL4LMs)
- [Reward Model Ensembles Help Mitigate Overoptimization](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dda7f9378a210c25e470e19304cce85d-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/tlc4418/llm_optimization)

<a name="noisy-verifiers" />

### <u>Noisy Verifiers & Rewards in RLVR</u>

- [Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards](https://openreview.net/forum?id=UY4F6bcSSP) (NeurIPS 2026)
- [From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning](https://arxiv.org/abs/2505.22203) (EMNLP 2026) [:octocat:](https://github.com/hkust-nlp/RL-Verifier-Robustness)
- [Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping](https://arxiv.org/abs/2608.24135) (EMNLP 2026)
- [Delay, Plateau, or Collapse: Evaluating the Impact of Systematic Verification Error on RLVR](https://arxiv.org/abs/2605.02909) (COLM 2026) [:octocat:](https://github.com/eth-sri/llm-verifier-noise)
- [Noise-corrected GRPO: From Noisy Rewards to Unbiased Gradients](https://openreview.net/forum?id=mnU8odBWYE) (ICML 2026) [:octocat:](https://github.com/omarito101/GRPO_project)
- [Rate or Fate? RLVεR: Reinforcement Learning with Verifiable Noisy Rewards](https://openreview.net/forum?id=LwB2EacVT6) (ICML 2026) [:octocat:](https://github.com/cognichip/Noisy-RL)
- [Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs](https://openreview.net/forum?id=SGUSUm2491) (ICML 2026) [:octocat:](https://github.com/idwts/How-RLVR-Activates-Memorization-Shortcuts)
- [Spurious Rewards: Rethinking Training Signals in RLVR](https://openreview.net/forum?id=tqTNOpkP5j) (ICML 2026) [:octocat:](https://github.com/ruixin31/Spurious_Rewards)
- [Can LLMs Learn to Reason Robustly under Noisy Supervision?](https://arxiv.org/abs/2604.03993) (arXiv 2026) [:octocat:](https://github.com/ShenzhiYang2000/OLR)
- [Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d99e8e80a6c41e148db686918dd7eab3-Abstract-Conference.html) (ICLR 2026)
- [FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fe5c31e525e9a26a1426ab0b589f42fe-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/yyDing1/FAPO)
- [HARDTESTGEN: A High-Quality RL Verifier Generation Pipeline for LLM Algorithimic Coding](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3c11cea35aa2e4797b2f239da378927b-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/LeiLiLab/HardTestGen)
- [LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking](https://arxiv.org/abs/2604.15149) (arXiv 2026)
- [When Can LLMs Learn to Reason with Weak Supervision?](https://arxiv.org/abs/2604.18574) (arXiv 2026)
- [NaDRO: Leveraging Dual-Reward Strategies for LLMs Training on Noisy Data](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d384ee6e65b5a10150b599a6d251663a-Abstract-Conference.html) (NeurIPS 2025)
- [Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers](https://arxiv.org/abs/2510.00915) (arXiv 2025) [:octocat:](https://github.com/caixq1996/noisy-RLVR)
- [Cooper: Co-Optimizing Policy and Reward Models in Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2508.05613) (arXiv 2025) [:octocat:](https://github.com/zju-real/cooper)
- [Mitigating Think-Answer Mismatch in LLM Reasoning Through Noise-Aware Advantage Reweighting](https://arxiv.org/abs/2508.05928) (arXiv 2025) [:octocat:](https://github.com/shenpeijun0212/S-GRPO)
- [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) (arXiv 2025)
- [The Climb Carves Wisdom Deeper Than the Summit: On the Noisy Rewards in Learning to Reason](https://arxiv.org/abs/2505.22653) (arXiv 2025) [:octocat:](https://github.com/trestad/Noisy-Rewards-in-Learning-to-Reason)
- [TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning](https://arxiv.org/abs/2505.14625) (arXiv 2025) [:octocat:](https://github.com/uw-nsl/TinyV)

<a name="imperfect-prm" />

### <u>Imperfect Process Reward Models</u>

- [Adversarial Training for Process Reward Models](https://openreview.net/forum?id=43Hwj59jZg) (ICML 2026)
- [DC-W2S: Dual-Consensus Weak-to-Strong Training for Reliable Process Reward Modeling in Biological Reasoning](https://openreview.net/forum?id=Lh3Hico2Pe) (ICML 2026)
- [Reward Under Attack: Analyzing the Robustness and Hackability of Process Reward Models](https://openreview.net/forum?id=phHnze3Nfm) (ICML 2026) [:octocat:](https://github.com/SqueezeAILab/reward-under-attack)
- [Towards Robust Process Reward Modeling via Noise-aware Learning](https://arxiv.org/abs/2601.12748) (arXiv 2026)
- [Know What You Don't Know: Uncertainty Calibration of Process Reward Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/37664246a1e07e212ddacea6e5a523f2-Abstract-Conference.html) (NeurIPS 2025)
- [SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4dd0cfda282c778a21b38fd1d637f37b-Abstract-Conference.html) (NeurIPS 2025)
- [Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/be91eb86eb74efc055cff83e953f86ce-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/CJReinforce/PURE)
- [The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://aclanthology.org/2025.findings-acl.547/) (ACL Findings 2025)

<a name="reward-uncertainty" />

### <u>Reward Uncertainty & Misspecification</u>

- [Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](https://openreview.net/forum?id=DfhMMHXDuu) (ICML 2026) [:octocat:](https://github.com/GuoweiRong/Bayesian-Non-negative-Reward-Model)
- [Ask a Strong LLM Judge when Your Reward Model is Uncertain](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6c312cf14321a99ae3788a41593cf2bd-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/horizon-llm/uncertainty-router)
- [Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective](https://proceedings.mlr.press/v267/huang25o.html) (ICML 2025) [:octocat:](https://github.com/jiaweihhuang/RLHF_RewardTransfer)
- [Policy Filtration for RLHF to Mitigate Noise in Reward Models](https://proceedings.mlr.press/v267/zhang25bq.html) (ICML 2025) [:octocat:](https://github.com/swtheing/PF-PPO-RLHF)
- [Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization via Chi-Squared Preference Optimization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e785d64baa2cde06c69d9b7a14636ef2-Abstract-Conference.html) (ICLR 2025)
- [Uncertainty and Influence aware Reward Model Refinement for Reinforcement Learning from Human Feedback](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fd7259e22add6de6df8ff0ccc902a34d-Abstract-Conference.html) (ICLR 2025)
- [Robust Preference Optimization through Reward Model Distillation](https://openreview.net/forum?id=E2zKNuwNDc) (TMLR 2025)
- [Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1a8189929f3d7bd6183718f42c3f4309-Abstract-Conference.html) (NeurIPS 2024)
- [Mitigating Reward Overoptimization via Lightweight Uncertainty Estimation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/94bbcb744bbada8808fda05b9d9290d6-Abstract-Conference.html) (NeurIPS 2024)
- [Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa69e968b7319fd42524febd41475fb3-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/YSLIU627/Regularized-Preference-Optimization)

<a name="weak-missing-supervision" />

## <u>Weak & Missing Supervision</u>

<a name="weak-to-strong" />

### <u>Weak-to-Strong Generalization</u>

- [The Mechanism of Weak-to-Strong Generalization: Feature Elicitation from Latent Knowledge](https://openreview.net/forum?id=2Ibq2ITNEb) (NeurIPS 2026)
- [Contrastive Weak-to-Strong Generalization](https://openreview.net/forum?id=maOM1OWNZ6) (ICML 2026) [:octocat:](https://github.com/jianghoucheng/ConG)
- [EnsemW2S: Enhancing Weak-to-Strong Generalization with Large Language Model Ensembles](https://aclanthology.org/2026.findings-acl.2093/) (ACL Findings 2026)
- [Generalizing Trust: Weak-to-Strong Trustworthiness in Language Models](https://aclanthology.org/2026.acl-long.2163/) (ACL 2026)
- [Removing Sandbagging in LLMs by Training with Weak Supervision](https://openreview.net/forum?id=J4DiM8EiMJ) (ICML 2026)
- [Trust Functions: Near Lossless Weak-to-Strong Generalization by Learning to Trust the Weak Teacher](https://openreview.net/forum?id=6bXLKr5NMT) (ICML 2026) [:octocat:](https://github.com/ardauzunoglu/trust-functions)
- [Does Weak-to-strong Generalization Happen under Spurious Correlations?](https://proceedings.iclr.cc/paper_files/paper/2026/hash/375a22d76e1c7d30c42bca9be47aab76-Abstract-Conference.html) (ICLR 2026)
- [Towards Scalable Oversight via Partitioned Human Supervision](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b6f03ce953cf7f07dafa20891d2539e3-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/R-Yin-217/Towards-Scalable-Oversight-via-Partitioned-Human-Supervision)
- [Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](https://proceedings.iclr.cc/paper_files/paper/2026/hash/650f33fdfaa83f432b9cfa2a2b45d663-Abstract-Conference.html) (ICLR 2026)
- [Weak-to-Strong Generalization with Failure Trajectories](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f3a935fc7a991a4c2be0606fb7f50098-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/yeruimeng/TraTree)
- [Incentivizing Strong Reasoning from Weak Supervision](https://aclanthology.org/2026.eacl-long.336/) (EACL 2026) [:octocat:](https://github.com/W2SR-ARR/Code)
- [Selective Weak-to-Strong Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/41089) (AAAI 2026)
- [Weak-to-Strong Generalization under Distribution Shifts](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e19583fd7768217b49db8f77c9f3e952-Abstract-Conference.html) (NeurIPS 2025)
- [Limitations of Refinement Methods for Weak to Strong Generalization](https://openreview.net/forum?id=OKvSnV5Ar7) (COLM 2025)
- [Discrepancies are Virtue: Weak-to-Strong Generalization through Lens of Intrinsic Dimension](https://proceedings.mlr.press/v267/dong25g.html) (ICML 2025)
- [Great Models Think Alike and this Undermines AI Oversight](https://proceedings.mlr.press/v267/goel25b.html) (ICML 2025) [:octocat:](https://github.com/model-similarity/lm-similarity)
- [How to Mitigate Overfitting in Weak-to-strong Generalization?](https://aclanthology.org/2025.acl-long.784/) (ACL 2025)
- [Relating Misfit to Gain in Weak-to-Strong Generalization Beyond the Squared Loss](https://proceedings.mlr.press/v267/mulgund25a.html) (ICML 2025) [:octocat:](https://github.com/abhmul/general-misfit-gain)
- [Revisiting Weak-to-Strong Generalization in Theory and Practice: Reverse KL vs. Forward KL](https://aclanthology.org/2025.findings-acl.148/) (ACL Findings 2025)
- [Weak-to-Strong Honesty Alignment via Learning-to-Rank Supervision](https://aclanthology.org/2025.findings-acl.529/) (ACL Findings 2025) [:octocat:](https://github.com/zewanfaan/WHAT_Honesty)
- [Bayesian WeakS-to-Strong from Text Classification to Generation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/495e55f361708bedbab5d81f92048dcd-Abstract-Conference.html) (ICLR 2025)
- [Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision](https://proceedings.iclr.cc/paper_files/paper/2025/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/helloelwin/iterative-label-refinement)
- [MACPO: Weak-to-Strong Alignment via Multi-Agent Contrastive Preference Optimization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9ddd47ee8d0b9543925a8db3e9d879b3-Abstract-Conference.html) (ICLR 2025)
- [Super(ficial)-alignment: Strong Models May Deceive Weak Models in Weak-to-Strong Generalization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/keven980716/weak-to-strong-deception)
- [Weak to Strong Generalization for Large Language Models with Multi-capabilities](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1ebcb8f051d0c178474619bbfb32b340-Abstract-Conference.html) (ICLR 2025)
- [Weak-to-Strong Generalization Through the Data-Centric Lens](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e44c4b5321605735be9761f9b758fc-Abstract-Conference.html) (ICLR 2025)
- [Weak-to-Strong Preference Optimization: Stealing Reward from Weak Aligned Model](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5beb3a846137dd6524f2da17c97d9426-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/zwhong714/weak-to-strong-preference-optimization)
- [Debate Helps Weak-to-Strong Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/34952) (AAAI 2025)
- [Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5b6346a05a537d4cdb2f50323452a9fe-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/Edward-Sun/easy-to-hard)
- [Quantifying the Gain in Weak-to-Strong Generalization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e4a0d8aef3567f742b0794844d9b5847-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/chogba/wtsg-regression)
- [Theoretical Analysis of Weak-to-Strong Generalization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5358d1e6138dff5718c5e9790f5fa593-Abstract-Conference.html) (NeurIPS 2024)
- [Weak-to-Strong Reasoning](https://aclanthology.org/2024.findings-emnlp.490/) (EMNLP Findings 2024) [:octocat:](https://github.com/GAIR-NLP/weak-to-strong-reasoning)
- [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](https://proceedings.mlr.press/v235/burns24b.html) (ICML 2024) [:octocat:](https://github.com/openai/weak-to-strong)

<a name="label-free-rl" />

### <u>Label-Free & Self-Rewarding RL</u>

- [CTRL: Continual Test-Time Reinforcement Learning for Large Language Models](https://openreview.net/forum?id=MBS7pBUFkB) (NeurIPS 2026)
- [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://openreview.net/forum?id=USnGWfYVux) (NeurIPS 2026) [:octocat:](https://github.com/Chengsong-Huang/G-Zero)
- [Reasoning Warm-up: Scaling Label-free RL via Verifiable Surrogate Rewards](https://openreview.net/forum?id=WTnplymile) (NeurIPS 2026)
- [SARL: Label-Free Reinforcement Learning by Rewarding Reasoning Topology](https://openreview.net/forum?id=z1jRrHayf1) (NeurIPS 2026)
- [SR-GRPO: Stable Rank as an Intrinsic Geometric Reward for Large Language Model Alignment](https://openreview.net/forum?id=eOONOlDp7u) (NeurIPS 2026)
- [Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL](https://openreview.net/forum?id=GPTwtahTVC) (NeurIPS 2026) [:octocat:](https://github.com/UCSB-AI/self-play-data-gating)
- [Tool Verification for Test-Time Reinforcement Learning](https://openreview.net/forum?id=Jd7P3rhNnN) (NeurIPS 2026)
- [Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework](https://openreview.net/forum?id=D8K5QK3kwT) (NeurIPS 2026)
- [Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning](https://arxiv.org/abs/2511.16043) (COLM 2026) [:octocat:](https://github.com/aiming-lab/Agent0)
- [Peer-Predictive Self-Training for Language Model Reasoning](https://arxiv.org/abs/2604.13356) (COLM 2026)
- [Beyond Majority Voting: Self-Reflective Test-Time Reinforcement Learning for LLM Reasoning](https://openreview.net/forum?id=kSuRY5sCuR) (ICML 2026)
- [Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning](https://aclanthology.org/2026.acl-long.1727/) (ACL 2026) [:octocat:](https://github.com/szu-tera/SCOPE)
- [Breaking the Self-Confirming Loop: Diagnosing and Mitigating Systemic Reward Bias in Self-Rewarding RL](https://openreview.net/forum?id=oAagVd30Fn) (ICML 2026)
- [CPC-GRPO: Answer-Free Reinforcement Learning with Cross-Prompt Consensus Rewards](https://aclanthology.org/2026.findings-acl.1486/) (ACL Findings 2026)
- [CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution](https://aclanthology.org/2026.acl-long.1376/) (ACL 2026) [:octocat:](https://github.com/ZJU-REAL/CoVerRL)
- [ECHO: Entropy-Confidence Hybrid Optimization for Test-Time Reinforcement Learning](https://openreview.net/forum?id=Zk0qjcqjV9) (ICML 2026)
- [Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs](https://aclanthology.org/2026.acl-long.797/) (ACL 2026)
- [On the Generalization Gap in Self-Evolving Language Model Reasoning](https://openreview.net/forum?id=mnUidYi5qO) (ICML 2026)
- [One-shot Entropy Minimization for Language Model Reasoning](https://openreview.net/forum?id=0yUoCFSUaY) (ICML 2026) [:octocat:](https://github.com/UbiquantAI/one-shot-em)
- [Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning](https://aclanthology.org/2026.findings-acl.1865/) (ACL Findings 2026) [:octocat:](https://github.com/yuyongcan/DDRL)
- [Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8587c39571f3e327725e2812749a9389-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/tmlr-group/Co-rewarding)
- [How Far Can Unsupervised RLVR Scale LLM Training?](https://proceedings.iclr.cc/paper_files/paper/2026/hash/18929f3253e4d4e28086a89ab88d1276-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/PRIME-RL/TTRL/tree/urlvr-dev)
- [Learning to Reason without External Rewards](https://proceedings.iclr.cc/paper_files/paper/2026/hash/052e2404709e8df9dc38fbc7bfec5b80-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/sunblaze-ucb/Intuitor)
- [Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ccf6d8b4a1fe9d9c8192f00c713872ea-Abstract-Conference.html) (ICLR 2026)
- [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d49b9aacebda61051166335af6fd3061-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/Chengsong-Huang/R-Zero)
- [RESTRAIN: From Spurious Votes to Signals — Self-Training RL with Self-Penalization](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0a2b005d2ea5dbd28331065427017c68-Abstract-Conference.html) (ICLR 2026)
- [Reinforcing General Reasoning Without Verifiers](https://proceedings.iclr.cc/paper_files/paper/2026/hash/71d207613c8eab1e42473143d3a598e0-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/sail-sg/VeriFree)
- [Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/86178c06fd59ebc58dc31f8edb3c54af-Abstract-Conference.html) (ICLR 2026)
- [Semantic Voting: A Self-Evaluation-Free Approach for Efficient LLM Self-Improvement on Unverifiable Open-ended Tasks](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3db3e1f192877e47bf48c93cae238e51-Abstract-Conference.html) (ICLR 2026)
- [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9837dc00ff67d176373268ed48042d49-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)
- [Beyond Verifiable Rewards: Scaling Reinforcement Learning in Language Models to Unverifiable Data](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6bd67a424dc59481e1e5a5061ffc8dfe-Abstract-Conference.html) (NeurIPS 2025)
- [Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/565f995643da6329cec701f26f8579f5-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/sastpg/CoVo)
- [Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization](https://proceedings.neurips.cc/paper_files/paper/2025/hash/614f8eba720cfc7ff00274bd64fb0a3f-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/QingyangZhang/EMPO)
- [SeRL: Self-play Reinforcement Learning for Large Language Models with Limited Data](https://proceedings.neurips.cc/paper_files/paper/2025/hash/95c2cbe23fb6d28a4ae908aa7f3de5bf-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/wantbook-book/SeRL)
- [TTRL: Test-Time Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/be690ea16f005c174f6c4102a5970e67-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/PRIME-RL/TTRL)
- [The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0d6da26625f780df1572cf7c8dc3faa6-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/1KE-JI/UPFT)
- [The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9a14cf2ee36cf2524f54d8f196a6734c-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/shivamag125/EM_PT)
- [Evolving Language Models without Labels: Majority Drives Selection, Novelty Promotes Variation](https://arxiv.org/abs/2509.15194) (arXiv 2025) [:octocat:](https://github.com/YujunZhou/EVOL-RL)
- [Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning](https://aclanthology.org/2025.acl-long.644/) (ACL 2025) [:octocat:](https://github.com/xufangzhi/Genius)
- [No Free Lunch: Rethinking Internal Feedback for LLM Reasoning](https://arxiv.org/abs/2506.17219) (arXiv 2025)
- [Unsupervised Elicitation of Language Models](https://arxiv.org/abs/2506.10139) (arXiv 2025) [:octocat:](https://github.com/Jiaxin-Wen/Unsupervised-Elicitation)
- [Can Large Reasoning Models Self-Train?](https://arxiv.org/abs/2505.21444) (arXiv 2025) [:octocat:](https://github.com/tajwarfahim/srt)
- [Maximizing Confidence Alone Improves Reasoning](https://arxiv.org/abs/2505.22660) (arXiv 2025)

<a name="partial-supervision" />

### <u>Partial, Negative & Incomplete Supervision</u>

- [MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels](https://openreview.net/forum?id=A5lXDI842J) (NeurIPS 2026)
- [Negative-Only Policy Optimization for One-Sided Verifiable Rewards](https://openreview.net/forum?id=RpadYP6yxY) (NeurIPS 2026)
- [Positive–Unlabeled Reinforcement Learning Distillation for On-Premise Small Models](https://openreview.net/forum?id=WyQlim30lN) (ICML 2026)
- [What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time](https://aclanthology.org/2026.acl-long.1337/) (ACL 2026) [:octocat:](https://github.com/Jasper-Yan/SCRL)
- [Annotation-Efficient Honesty Alignment via Confidence Elicitation and Calibration](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e9ce8dc03e403c1198ac896fcfdcee0-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/Trustworthy-Information-Access/Annotation-Efficient-Universal-Honesty-Alignment)
- [TraPO: A Semi-Supervised Reinforcement Learning Framework for Boosting LLM Reasoning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bb290ccaacc433da7ec192e6c4d909b3-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/ShenzhiYang2000/TRAPO)
- [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b8026fe01f5f8f7a80a1a9bf01d08cad-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/TianHongZXY/RLVR-Decomposed)
- [Free Process Rewards without Process Labels](https://proceedings.mlr.press/v267/yuan25c.html) (ICML 2025) [:octocat:](https://github.com/PRIME-RL/ImplicitPRM)
- [Semi-supervised Fine-tuning for Large Language Models](https://aclanthology.org/2025.findings-naacl.151/) (NAACL Findings 2025) [:octocat:](https://github.com/luo-junyu/SemiEvol)
- [Varying Shades of Wrong: Aligning LLMs with Wrong Answers Only](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1aa1fde3661b23ba9b043082069fd144-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/yaojh18/Varying-Shades-of-Wrong)
- [RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4b77d5b896c321a29277524a98a50215-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/ars22/scaling-LLM-math-synthetic-data)

<a name="pseudo-labels" />

### <u>Learning from Pseudo-Labels</u>

- [Scaling Reward Modeling without Human Supervision](https://openreview.net/forum?id=yD1cdr9cGj) (NeurIPS 2026)
- [ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](https://aclanthology.org/2026.acl-long.1830/) (ACL 2026)
- [Self-Training Large Language Models with Confident Reasoning](https://aclanthology.org/2025.findings-emnlp.806/) (EMNLP Findings 2025) [:octocat:](https://github.com/hsjang0/CORE-PO)
- [Self-Consistency Preference Optimization](https://proceedings.mlr.press/v267/prasad25a.html) (ICML 2025)
- [Learning with Less: Knowledge Distillation from Large Language Models via Unlabeled Data](https://aclanthology.org/2025.findings-naacl.142/) (NAACL Findings 2025)
- [Preference Optimization for Reasoning with Pseudo Feedback](https://proceedings.iclr.cc/paper_files/paper/2025/hash/31a57804448363bcab777f818f75f5b4-Abstract-Conference.html) (ICLR 2025)

<a name="imperfect-sft-distillation-data" />

## <u>Imperfect SFT & Distillation Data</u>

<a name="noisy-sft" />

### <u>Noisy Labels in Fine-Tuning</u>

- [Analyzing the Effect of Noise in LLM Fine-tuning](https://arxiv.org/abs/2604.12469) (arXiv 2026)
- [Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6e0a771a0ca0bc5ff0a068f76557cc3d-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/LemonTea03/XTF)
- [REALM: Reliable Expertise-Aware Language Model Fine-Tuning from Noisy Annotations](https://arxiv.org/abs/2604.17289) (arXiv 2026)
- [Why LoRA Resists Label Noise: A Theoretical Framework for Noise-Robust Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2602.00084) (arXiv 2026)
- [Noise-Robustness Through Noise: A Framework combining Asymmetric LoRA with Poisoning MoE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a204aa68ab4e970e1ceccfb5b5cdc5e4-Abstract-Conference.html) (NeurIPS 2025)
- [Self Iterative Label Refinement via Robust Unlabeled Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2d3569e4d2ca7e8edeed31a09f817bd9-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/HikaruAsano/self-iterative-label-refinement)
- [Identifying Noise in Human-Created Datasets using Training Dynamics from Generative Models](https://aclanthology.org/2025.findings-emnlp.840/) (EMNLP Findings 2025)
- [Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning](https://proceedings.mlr.press/v267/pang25a.html) (ICML 2025) [:octocat:](https://github.com/UCSC-REAL/TokenCleaning)
- [Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning](https://aclanthology.org/2025.findings-acl.792/) (ACL Findings 2025)
- [FiNE: Filtering and Improving Noisy Data Elaborately with Large Language Models](https://aclanthology.org/2025.naacl-long.437/) (NAACL 2025) [:octocat:](https://github.com/jlhe2000/FiNE)
- [Improving Data Efficiency via Curating LLM-Driven Rating Systems](https://proceedings.iclr.cc/paper_files/paper/2025/hash/faa6144674bce872365874c576b4f56f-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/UCSC-REAL/DS2)
- [Towards Robust and Generalized Parameter-Efficient Fine-Tuning for Noisy Label Learning](https://aclanthology.org/2024.acl-long.322/) (ACL 2024) [:octocat:](https://github.com/yeachan-kr/clear)

<a name="flawed-reasoning-data" />

### <u>Flawed Demonstrations & Reasoning Traces</u>

- [Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning](https://aclanthology.org/2026.acl-long.74/) (ACL 2026) [:octocat:](https://github.com/Tim-Siu/reinforcement-distillation)
- [Learning from Mistakes: Negative Reasoning Samples Enhance Out-of-Domain Generalization](https://aclanthology.org/2026.acl-long.1370/) (ACL 2026)
- [RIFT: Repurposing Negative Samples via Reward-Informed Fine-Tuning](https://aclanthology.org/2026.findings-acl.706/) (ACL Findings 2026)
- [Language Models Can Easily Learn to Reason from Demonstrations](https://aclanthology.org/2025.findings-emnlp.866/) (EMNLP Findings 2025)
- [Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning](https://aclanthology.org/2025.findings-emnlp.57/) (EMNLP Findings 2025)
- [LEMMA: Learning from Errors for MatheMatical Advancement in LLMs](https://aclanthology.org/2025.findings-acl.605/) (ACL Findings 2025) [:octocat:](https://github.com/pzs19/LEMMA)
- [Learning from others' mistakes: Finetuning machine translation models with span-level error annotations](https://proceedings.mlr.press/v267/zhang25p.html) (ICML 2025)
- [Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c239bac713017b0b2257b7622bf8aab3-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/facebookresearch/iGSM)
- [Self-Training Meets Consistency: Improving LLMs' Reasoning with Consistency-Driven Rationale Evaluation](https://aclanthology.org/2025.naacl-long.528/) (NAACL 2025) [:octocat:](https://github.com/jaehyeoklee-119/crest)
- [Can LLMs Learn From Mistakes? An Empirical Study on Reasoning Tasks](https://aclanthology.org/2024.findings-emnlp.46/) (EMNLP Findings 2024) [:octocat:](https://github.com/microsoft/LEMA)
- [Can LLMs Learn from Previous Mistakes? Investigating LLMs' Errors to Boost for Reasoning](https://aclanthology.org/2024.acl-long.169/) (ACL 2024) [:octocat:](https://github.com/YookiTong/Learn-from-Mistakes-CotErrorSet)
- [Gaining Wisdom from Setbacks: Aligning Large Language Models via Mistake Analysis](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fe24df54d5ccd95cdf830a309f2bf5c0-Abstract-Conference.html) (ICLR 2024)
- [Turning Dust into Gold: Distilling Complex Reasoning Capabilities from LLMs by Leveraging Negative Data](https://ojs.aaai.org/index.php/AAAI/article/view/29821) (AAAI 2024) [:octocat:](https://github.com/Yiwei98/TDG)

<a name="imperfect-teacher" />

### <u>Imperfect Teachers in Distillation</u>

- [Behavior Cloning is Not All You Need: The Optimality of On-Policy Distillation for Noisy Expert Feedback](https://openreview.net/forum?id=gLfGap1Y0U) (NeurIPS 2026)
- [Distilling What Matters: Confidence-Aware Selective Distillation for Large Language Models](https://openreview.net/forum?id=mhRs8gO4jl) (NeurIPS 2026)
- [When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning](https://openreview.net/forum?id=wPMiko3NFZ) (NeurIPS 2026) [:octocat:](https://github.com/SaFo-Lab/PW-OPSD)
- [Exploring Knowledge Purification in Multi-Teacher Knowledge Distillation for LLMs](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bee5a0651c7052e61771b9ff7b9f93d2-Abstract-Conference.html) (ICLR 2026)
- [Learning from Committee: Reasoning Distillation from a Mixture of Teachers with Peer-Review](https://aclanthology.org/2025.findings-acl.217/) (ACL Findings 2025) [:octocat:](https://github.com/zhuochunli/Learn-from-Committee)
- [On Teacher Hacking in Language Model Distillation](https://proceedings.mlr.press/v267/tiapkin25a.html) (ICML 2025)
- [Learning from Imperfect Data: Towards Efficient Knowledge Distillation of Autoregressive Language Models for Text-to-SQL](https://aclanthology.org/2024.findings-emnlp.403/) (EMNLP Findings 2024)
- [Teaching-Assistant-in-the-Loop: Improving Knowledge Distillation from Imperfect Teacher Models in Low-Budget Scenarios](https://aclanthology.org/2024.findings-acl.17/) (ACL Findings 2024)
- [Impossible Distillation for Paraphrasing and Summarization: How to Make High-quality Lemonade out of Small, Low-quality Model](https://aclanthology.org/2024.naacl-long.250/) (NAACL 2024) [:octocat:](https://github.com/jaehunjung1/impossible-distillation)

<a name="synthetic-data" />

### <u>Synthetic Data, Self-Improvement & Model Collapse</u>

- [From Collapse to Improvement: Statistical Perspectives on the Evolutionary Dynamics of Iterative Training on Contaminated Sources](https://openreview.net/forum?id=yfGxUZj1zT) (NeurIPS 2026)
- [No Model Required: Text Entropy Rate Filtering Prevents Iterative Fine-Tuning Collapse](https://openreview.net/forum?id=1rTomFsjg0) (NeurIPS 2026) [:octocat:](https://github.com/lewismath/no-model-required)
- [Self-Trained Verification for Training- and Test-Time Self-Improvement](https://openreview.net/forum?id=KdMT1obOmV) (NeurIPS 2026) [:octocat:](https://github.com/ar-forum/stv)
- [Language Generation with Replay: A Learning-Theoretic View of Model Collapse](https://openreview.net/forum?id=scnRgI2hhX) (ICML 2026)
- [Learning from Synthetic Data without Model Collapse in Iterative Instruction Tuning](https://arxiv.org/abs/2607.17043) (arXiv 2026)
- [Observations and Remedies for Large Language Model Bias in Self-Consuming Performative Loop](https://aclanthology.org/2026.acl-long.1561/) (ACL 2026) [:octocat:](https://github.com/UCSC-REAL/SCPL)
- [Position: Multiple Definitions & Unrealistic Assumptions of Model Collapse Distract from Real World Threats](https://icml.cc/virtual/2026/poster/67117) (ICML 2026)
- [Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence](https://proceedings.iclr.cc/paper_files/paper/2026/hash/29806b2e4dcb35e42abed21d43e2b7d8-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/liuqiyuanhhh/Verified-Synthetic-Data)
- [What happens when generative AI models train recursively on each others' outputs?](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e5e38389c8c21a151ea898f877be9f6c-Abstract-Conference.html) (ICLR 2026)
- [Escaping Collapse: The Strength of Weak Data for Large Language Model Training](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2a246ec4437808525a088abae744c2d3-Abstract-Conference.html) (NeurIPS 2025)
- [Self-Verification Provably Prevents Model Collapse in Recursive Synthetic Training](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3380e8116452e0efbf36f35d95e88c94-Abstract-Conference.html) (NeurIPS 2025)
- [Machine-generated text detection prevents language model collapse](https://aclanthology.org/2025.emnlp-main.1506/) (EMNLP 2025) [:octocat:](https://github.com/GeorgeDrayson/model_collapse)
- [Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?](https://aclanthology.org/2025.emnlp-main.1643/) (EMNLP 2025)
- [Collapse or Thrive: Perils and Promises of Synthetic Data in a Self-Generating World](https://proceedings.mlr.press/v267/kazdan25a.html) (ICML 2025) [:octocat:](https://github.com/RylanSchaeffer/KoyejoLab-Collapse-or-Thrive)
- [How to Synthesize Text Data without Model Collapse?](https://proceedings.mlr.press/v267/zhu25d.html) (ICML 2025) [:octocat:](https://github.com/Xuekai-Zhu/toedit)
- [Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification](https://proceedings.iclr.cc/paper_files/paper/2025/hash/df2d62b96a4003203450cf89cd338bb7-Abstract-Conference.html) (ICLR 2025)
- [Bias Amplification in Language Model Evolution: An Iterated Learning Perspective](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4418f6a54f4314202688d77956e731ce-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/Joshua-Ren/iICL)
- [Unveiling the Flaws: Exploring Imperfections in Synthetic Data and Mitigation Strategies for Large Language Models](https://aclanthology.org/2024.findings-emnlp.873/) (EMNLP Findings 2024)
- [How bad is training on synthetic data? A statistical analysis of language model collapse](https://openreview.net/forum?id=t3z6UlV09o) (COLM 2024)
- [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://openreview.net/forum?id=5B2K4LRgmz) (COLM 2024)
- [A Tale of Tails: Model Collapse as a Change of Scaling Laws](https://proceedings.mlr.press/v235/dohmatob24b.html) (ICML 2024)

<a name="imperfect-trajectories" />

### <u>Imperfect Agent Trajectories</u>

- [Failures are Treasures: Constructing a Pedagogical Bridge for Agentic Strategy Distillation](https://aclanthology.org/2026.findings-acl.938/) (ACL Findings 2026)
- [Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs](https://openreview.net/forum?id=KaTYG9LGJv) (ICML 2026) [:octocat:](https://github.com/haruhi-sudo/SCoRe)
- [STeCa: Step-level Trajectory Calibration for LLM Agent Learning](https://aclanthology.org/2025.findings-acl.604/) (ACL Findings 2025) [:octocat:](https://github.com/WangHanLinHenry/STeCa)
- [Teaching Text Agents to Learn Sequential Decision Making from Failure](https://aclanthology.org/2025.acl-long.1526/) (ACL 2025) [:octocat:](https://github.com/riken-grp/text-agent)
- [Exploring Expert Failures Improves LLM Agent Tuning](https://arxiv.org/abs/2504.13145) (arXiv 2025) [:octocat:](https://github.com/lan-lc/EEF)
- [NAT: Enhancing Agent Tuning with Negative Samples](https://aclanthology.org/2025.naacl-long.378/) (NAACL 2025) [:octocat:](https://github.com/Reason-Wang/NAT)
- [Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training](https://arxiv.org/abs/2501.11425) (arXiv 2025) [:octocat:](https://github.com/ByteDance-Seed/Agent-R)
- [Advancing Tool-Augmented Large Language Models: Integrating Insights from Errors in Inference Trees](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c0f7ee1901fef1da4dae2b88dfd43195-Abstract-Conference.html) (NeurIPS 2024)
- [Re-ReST: Reflection-Reinforced Self-Training for Language Agents](https://aclanthology.org/2024.emnlp-main.861/) (EMNLP 2024) [:octocat:](https://github.com/PlusLabNLP/Re-ReST)
- [Learn from Failure: Fine-tuning LLMs with Trial-and-Error Data for Intuitionistic Propositional Logic Proving](https://aclanthology.org/2024.acl-long.45/) (ACL 2024) [:octocat:](https://github.com/ucsd-atp/PropL)
- [Trial and Error: Exploration-Based Trajectory Optimization of LLM Agents](https://aclanthology.org/2024.acl-long.409/) (ACL 2024) [:octocat:](https://github.com/Yifan-Song793/ETO)

<a name="corrupted-feedback-theory" />

## <u>Corrupted Feedback & Theory</u>

<a name="poisoned-feedback" />

### <u>Poisoned Preference & Reward Data</u>

- [BiasTrojan: LLM Judgers Are Easily Distorted by Few Hundreds of Contrastive Biased Training Data](https://openreview.net/forum?id=k8KvasBchQ) (NeurIPS 2026)
- [Few-Shot Truly Benign DPO Attack for Jailbreaking LLMs](https://openreview.net/forum?id=e0AAyKxXc8) (NeurIPS 2026)
- [GREAT: Generalizable Backdoor Attacks in RLHF via Emotion-Aware Trigger Synthesis](https://aclanthology.org/2026.findings-emnlp.1737/) (EMNLP Findings 2026) [:octocat:](https://github.com/PiyushWithPant/GREAT)
- [Cordyceps: Covert Control Attacks on LLMs via Data Poisoning](https://www.usenix.org/conference/usenixsecurity26/presentation/shao-zedian) (USENIX Security 2026) [:octocat:](https://github.com/Sadcardation/cordyceps)
- [Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases](https://openreview.net/forum?id=qNsrhcCe4y) (ICML 2026)
- [Backdoors in RLVR: Jailbreak Backdoors in LLMs From Verifiable Reward](https://aclanthology.org/2026.acl-long.1484/) (ACL 2026) [:octocat:](https://github.com/yuki-younai/Backdoor_in_RLVR)
- [Efficient Preference Poisoning Attack on Offline RLHF](https://openreview.net/forum?id=32XNOcwH1z) (ICML 2026)
- [Sequential Data Poisoning in LLM Post-Training](https://arxiv.org/abs/2606.04929) (arXiv 2026) [:octocat:](https://github.com/jcksanderson/sequential-poisoning)
- [Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment](https://arxiv.org/abs/2511.09105) (AAAI 2026) [:octocat:](https://github.com/akimotolab/PoisoningCostMinimization)
- [Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e6c5195dac675f03d0fcf3955bcdd3c9-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/liangzid/VirusInfectionAttack)
- [PoisonBench: Assessing Language Model Vulnerability to Poisoned Preference Data](https://proceedings.mlr.press/v267/fu25e.html) (ICML 2025) [:octocat:](https://github.com/TingchenFu/PoisonBench)
- [Policy Teaching via Data Poisoning in Learning from Human Preferences](https://proceedings.mlr.press/v258/nika25a.html) (AISTATS 2025)
- [Preference Poisoning Attacks on Reward Model Learning](https://arxiv.org/abs/2402.01920) (IEEE S&P 2025)
- [BadJudge: Backdoor Vulnerabilities of LLM-As-A-Judge](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2e48f562a2c8f64c7404a6c3a518af74-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/TerryTong-Git/badjudge)
- [LLM Misalignment via Adversarial RLHF Platforms](https://arxiv.org/abs/2503.03039) (arXiv 2025)
- [Is Poisoning a Real Threat to DPO? Maybe More So Than You Think](https://ojs.aaai.org/index.php/AAAI/article/view/34968) (AAAI 2025) [:octocat:](https://github.com/pankayaraj/AAAI_2025_RLHFPoisoning)
- [Best-of-Venom: Attacking RLHF by Injecting Poisoned Preference Data](https://openreview.net/forum?id=v74mJURD1L) (COLM 2024)
- [RLHFPoison: Reward Poisoning Attack for Reinforcement Learning with Human Feedback in Large Language Models](https://aclanthology.org/2024.acl-long.140/) (ACL 2024)
- [Backdooring Instruction-Tuned Large Language Models with Virtual Prompt Injection](https://aclanthology.org/2024.naacl-long.337/) (NAACL 2024)
- [Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models](https://aclanthology.org/2024.naacl-long.171/) (NAACL 2024)
- [Universal Jailbreak Backdoors from Poisoned Human Feedback](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d1e63f4027efc7c44bfc253eafc15a58-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/ethz-spylab/rlhf-poisoning)

<a name="corruption-robust" />

### <u>Corruption-Robust Post-Training</u>

- [GradSentry: Gradient Spectral Entropy for Backdoor Sample Filtering in Large Language Model Fine-Tuning](https://aclanthology.org/2026.emnlp-main.210/) (EMNLP 2026)
- [Robust Direct Preference Optimization under Preference Label-Flip Attacks](https://arxiv.org/abs/2610.24945) (arXiv 2026)
- [Improved Bounds for Private and Robust Alignment](https://openreview.net/forum?id=r3PKGNlKET) (ICML 2026)
- [Merging Triggers, Breaking Backdoors: Defensive Poisoning for Instruction-Tuned Language Models](https://aclanthology.org/2026.acl-long.1113/) (ACL 2026) [:octocat:](https://github.com/mountinyy/MB-Defense)
- [Corruption-robust Offline Multi-agent Reinforcement Learning from Human Feedback](https://proceedings.mlr.press/v300/nika26a.html) (AISTATS 2026)
- [Towards Poisoning Robustness Certification for Natural Language Generation](https://arxiv.org/abs/2602.09757) (arXiv 2026)
- [Strategyproof Reinforcement Learning from Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2025/hash/92f43b1d33fae4aa1958f75317f0cec1-Abstract-Conference.html) (NeurIPS 2025)
- [Provably Mitigating Corruption, Overoptimization, and Verbosity Simultaneously in Offline and Online RLHF/DPO Alignment](https://arxiv.org/abs/2510.05526) (arXiv 2025)
- [A Unified Theoretical Analysis of Private and Robust Offline Alignment: from RLHF to DPO](https://proceedings.mlr.press/v267/zhou25ac.html) (ICML 2025)
- [Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning Attack](https://proceedings.mlr.press/v267/huang25b.html) (ICML 2025) [:octocat:](https://github.com/git-disl/Antidote)
- [Nearly Optimal Algorithms for Contextual Dueling Bandits from Adversarial Feedback](https://proceedings.mlr.press/v267/di25a.html) (ICML 2025)
- [SquareχPO: Differentially Private and Robust χ2-Preference Optimization in Offline Direct Alignment](https://proceedings.mlr.press/v267/zhou25ad.html) (ICML 2025)
- [Corruption Robust Offline Reinforcement Learning with Human Feedback](https://proceedings.mlr.press/v258/mandal25b.html) (AISTATS 2025)
- [Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/git-disl/Booster)
- [Learning from Imperfect Human Feedback: A Tale from Corruption-Robust Dueling](https://proceedings.iclr.cc/paper_files/paper/2025/hash/496b549556509bbb9770bf9d335c5800-Abstract-Conference.html) (ICLR 2025)
- [Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bcfdaf04b54a69f47623c973c864ee8d-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/git-disl/Lisa)
- [Robust Reinforcement Learning from Corrupted Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e04101138a3c94544760c1dbdf2c7a2d-Abstract-Conference.html) (NeurIPS 2024)
- [Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack](https://proceedings.neurips.cc/paper_files/paper/2024/hash/873c86d9a979ab80d8e2919510d4446b-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/git-disl/Vaccine)

<a name="theory" />

### <u>Theory</u>

- [Explaining and Preventing Alignment Collapse in Iterative RLHF](https://openreview.net/forum?id=3azE5mTgC1) (NeurIPS 2026) [:octocat:](https://github.com/GauthierE/fpo)
- [Finding the Signal in the Spam: Jointly Learning Rewards and Worker Reliability from Pairwise Comparisons](https://proceedings.mlr.press/v337/shejole26a.html) (UAI 2026) [:octocat:](https://github.com/KaustubhShejole/BoRa_EM)
- [Statistical Impossibility and Possibility of Aligning LLMs with Human Preferences: From Condorcet Paradox to Nash Equilibrium](https://icml.cc/virtual/2026/poster/71805) (ICML 2026)
- [Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO](https://openreview.net/forum?id=sg94PRd3kD) (ICML 2026) [:octocat:](https://github.com/srzer/Gap-in-Preference-Learning)
- [Learning Correlated Reward Models: Statistical Barriers and Opportunities](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d019cc362b0a179401f169c2d31b0989-Abstract-Conference.html) (ICLR 2026)
- [Robust Optimization for Mitigating Reward Hacking with Correlated Proxies](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0904c7edde20d7134a77fc7f9cd86ea2-Abstract-Conference.html) (ICLR 2026)
- [Theoretical Modeling of Large Language Model Self-Improvement Training Dynamics Through Solver-Verifier Gap](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1f6100363156cced8633f4e89dd8ceb1-Abstract-Conference.html) (ICLR 2026)
- [Why DPO is a Misspecified Estimator and How to Fix It](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8f6c674a7edf57912f1a5830f4a66e3f-Abstract-Conference.html) (ICLR 2026)
- [What Makes a Reward Model a Good Teacher? An Optimization Perspective](https://proceedings.neurips.cc/paper_files/paper/2025/hash/554e056fe2b6d9fd27ffcd3367ae1267-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/princeton-pli/what-makes-good-rm)
- [When Can Proxies Improve the Sample Complexity of Preference Learning?](https://proceedings.mlr.press/v267/zhu25f.html) (ICML 2025)
- [A Theoretical Framework for Partially-Observed Reward States in RLHF](https://proceedings.iclr.cc/paper_files/paper/2025/hash/38e7f39f0187b4a334ade462ecc33e96-Abstract-Conference.html) (ICLR 2025)
- [Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking](https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbe2cfe4767f3255160b73a36ae3162e-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/cassidylaidlaw/orpo)
- [Strong Preferences Affect the Robustness of Preference Models and Value Alignment](https://proceedings.iclr.cc/paper_files/paper/2025/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html) (ICLR 2025)
- [When Your AIs Deceive You: Challenges of Partial Observability in Reinforcement Learning from Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a995960dd0193654d6b18eca4ac5b936-Abstract-Conference.html) (NeurIPS 2024)
- [Goodhart's Law in Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6ad68a54eaa8f9bf6ac698b02ec05048-Abstract-Conference.html) (ICLR 2024)
<!-- papers:end -->
