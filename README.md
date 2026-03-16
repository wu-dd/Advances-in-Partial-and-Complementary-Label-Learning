# Advances in Partial/Complementary Label Learning 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/> <img src="https://img.shields.io/badge/Number%20of%20Items-213-FF6F00" alt="PaperNum"/> ![Stars](https://img.shields.io/github/stars/wu-dd/Advances-in-Partial-and-Complementary-Label-Learning?color=yellow&label=Stars) ![Forks](https://img.shields.io/github/forks/wu-dd/Advances-in-Partial-and-Complementary-Label-Learning?color=green&label=Forks)





**Advances in Partial/Complementary Label Learning** provides the most advanced and detailed information on partial/complementary label learning field.

**Partial/complementary label learning** is an emerging framework in weakly supervised machine learning with broad application prospects. It handles the case in which each training example corresponds to a candidate label set and only one label concealed in the set is the ground-truth label.

**This project is curated and maintained by [Dong-Dong Wu](https://wu-dd.github.io/)**. I will do my best to keep the project up to date. If you have any suggestions or are interested in being contributors, feel free to drop me an email.



#### [How to submit a pull request?](./CONTRIBUTING.md)

+ :globe_with_meridians: Project Page
+ :octocat: Code
+ :book: `bibtex`

## Latest Updates
+ [2026/03/16] Conducted another venue-by-venue and journal-by-journal semantic search (Google Scholar / Semantic Scholar / arXiv and venue pages), and added 20+ missing papers across NeurIPS/AAAI/TPAMI/PR/arXiv tracks (e.g., ComRank, What Makes PLL Effective, CausalPLL, Fast-MIPL, PR 2026 semi-supervised PLL papers, and several new arXiv 2025 PLL/partial-multi-label works).
+ [2025/09/22] Update some papers published on conferences. If you find some errors, please feel free to contact me!
+ [2024/12/08] Update some code links of papers.
+ [2024/09/13] A major overhaul of the original github repository. 

## Contents
- [Main](#main)
  - **Foundations & Surveys**
    - [Early Work](#early-work)
    - [Survey](#survey)
    - [Benchmark](#benchmark)
    - [Generative Modeling](#theory-based)
    - [Understanding](#understanding)
  - **Algorithms & Optimization**
    - [Better Optimization](#better-optimization)
    - [Dimensionality Reduction](#dimensionality-reduction)
    - [Negative Learning](#negative-learning)
  - **Problem Settings & Extensions**
    - [Partial Multi-Label Learning](#partial-multi-label-learning)
    - [Noisy Partial Label Learning](#noisy-partial-label-learning)
    - [Semi-Supervised Partial Label Learning](#semi-supervised-partial-label-learning)
    - [Multi-Instance Partial Label Learning](#multi-instance-learning)
    - [Imbalanced Partial Label Learning](#class-imbalance-problem)
    - [Out-of-Distribution Partial Label Learning](#ood-pll)
    - [Federated Partial Label Learning](#fed-pll)
    - [Partial Label Regression](#partial-label-regression)
    - [Multi-Complementary Label Learning](#multi-complementary-label-learning)
    - [Multi-View Learning](#multi-view-learning)
    - [Multi-Dimensional](#multi-dimensional)
  - **Learning Paradigms & Strategies**
    - [Adversarial Training](#adversarial-training)
    - [Incremental Learning](#incremental-learning)
    - [Online Learning](#online-learning)
    - [Domain Adaptation](#domain-adaptation)
    - [Few-Shot Learning](#few-shot-learning)
    - [Open-Set Problem](#open-set-problem)
    - [Data Augmentation](#data-augmentation)
    - [Conformal Prediction](#conformal-prediction)
- [Applications](#applications)
  - [Audio](#audio)
  - [Text](#text)
  - [Sequence](#sequence-classification)
  - [Recognition](#recognition)
  - [Object Localization](#object-localization)
  - [Map Reconstruction](#map-reconstruction)
  - [Semi-Supervised Learning](#semi-supervised-learning)
  - [Active Learning](#active-learning)
  - [Noisy Label Learning](#noisy-label-learning)
  - [Test-Time Adaptation](#TTA)
  - [Weak Supervision Evaluation](#weak-supervision-evaluation)

<a name="main" />

## <u>Main</u>

- [Learning from Complementary Labels](https://arxiv.org/abs/1705.07541) (NeurIPS 2017) [:octocat:](https://github.com/takashiishida/comp)
- [Learning from Partial Labels](https://jmlr.org/papers/v12/cour11a.html) (JMLR 2011)

<a name="foundations-and-surveys" />

### <u>Foundations & Surveys</u>

<a name="early-work" />

### <u>Early Work</u>
+ To be continued.

<a name="survey" />

### <u>Survey</u>

- [Partial label learning: Taxonomy, analysis and outlook](https://www.sciencedirect.com/science/article/abs/pii/S0893608023000825) (NN 2023)

<a name="benchmark" />

### <u>Benchmark</u>

- [CLImage: Human-Annotated Datasets for Complementary-Label Learning](https://arxiv.org/abs/2305.08295) (TMLR 2025)
- [Realistic Evaluation of Deep Partial-Label Learning Algorithms](https://arxiv.org/pdf/2502.10184) (ICLR 2025) [:octocat:](https://github.com/wwangwitsel/PLENCH)

<a name="theory-based" />

### <u>Generative Modeling</u>

- [Learning with Complementary Labels Revisited: The Selected-Completely-at-Random Setting Is More Practical](https://arxiv.org/abs/2311.15502) (ICML 2024) [:octocat:](https://github.com/wwangwitsel/SCARCE)
- [Unbiased Risk Estimator to Multi-Labeled Complementary Label Learning](https://www.ijcai.org/proceedings/2023/415) (IJCAI 2023) [:octocat:](https://github.com/GaoYi439/GDF)
- [Towards Unbiased Exploration in Partial Label Learning](https://arxiv.org/abs/2307.00465) (JMLR 2024)
- [Leveraged Weighted Loss for Partial Label Learning](https://arxiv.org/abs/2106.05731) (ICML 2021) [:octocat:](https://github.com/hongwei-wen/LW-loss-for-partial-label)
- [A Generative Model for Partial Label Learning](https://ieeexplore.ieee.org/abstract/document/9428103) (ICME 2021)
- [Provably Consistent Partial-Label Learning](https://arxiv.org/abs/2007.08929) (NeurIPS 2020) [:octocat:](https://lfeng1995.github.io/Code/RCCC.zip)
- [Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels](https://arxiv.org/abs/2007.02235) (ICML 2020)
- [Complementary-Label Learning for Arbitrary Losses and Models](https://arxiv.org/abs/1810.04327) (ICML 2019) [:octocat:](https://github.com/takashiishida/comp)
- [Learning with Biased Complementary Labels](https://arxiv.org/abs/1711.09535)  (ECCV 2018) [:octocat:](https://github.com/takashiishida/comp)

<a name="understanding" />

### <u>Understanding</u>

- [Rethinking Self-Distillation: Label Averaging and Enhanced Soft Label Refinement with Partial Labels](https://arxiv.org/abs/2402.10482) (ICLR 2025)
- [Realistic Evaluation of Deep Partial-Label Learning Algorithms](https://arxiv.org/abs/2502.10184) (ICLR 2025) [:octocat:](https://github.com/wwangwitsel/PLENCH)
- [Candidate Label Set Pruning: A Data-centric Perspective for Deep Partial-label Learning](https://openreview.net/forum?id=Fk5IzauJ7F) (ICLR 2024)
- [A Unifying Probabilistic Framework for Partially Labeled Data Learning](https://ieeexplore.ieee.org/document/9983986) (TPAMI 2023)
- [On the Power of Deep but Naive Partial Label Learning](https://arxiv.org/abs/2010.11600) (ICASSP 2021) [:octocat:](https://github.com/mikigom/DNPL-PyTorch)
- [Learning from a Complementary-label Source Domain: Theory and Algorithms](https://arxiv.org/abs/2008.01454) (TNNLS 2021)
- [Bridging Ordinary-Label Learning and Complementary-Label Learning](https://proceedings.mlr.press/v129/katsura20a.html) (ACML 2020)

<a name="algorithms-and-optimization" />

### <u>Algorithms & Optimization</u>

<a name="better-optimization" />

### <u>Better Optimization</u>

- [ComRank: Ranking Loss for Multi-Label Complementary Label Learning](https://neurips.cc/virtual/2025/poster/115492) (NeurIPS 2025) [:octocat:](https://github.com/JellyJamZhu/ComRank)
- [What Makes Partial-Label Learning Algorithms Effective?](https://palm.seu.edu.cn/zhangml/files/NeurIPS%2724e.pdf) (NeurIPS 2024)
- [Diffusion Disambiguation Models for Partial Label Learning](https://arxiv.org/abs/2507.00411) (arXiv 2025)
- [Tuning the Right Foundation Models is What You Need for Partial Label Learning](https://arxiv.org/abs/2506.05027) (arXiv 2025)
- [Revising the Problem of Partial Labels from the Perspective of CNNs' Robustness](https://arxiv.org/abs/2407.17630) (arXiv 2024)
- [Pre-Trained Vision-Language Models as Partial Annotators](https://arxiv.org/abs/2406.18550) (arXiv 2024)
- [Mixed Blessing: Class-Wise Embedding Guided Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2412.05029) (arXiv 2024)
- [Learning from Stochastic Labels](https://arxiv.org/abs/2302.00299) (ICME 2025)
- [Reduction-based Pseudo-label Generation for Instance-dependent Partial Label Learning](https://arxiv.org/abs/2410.20797) (arXiv 2024)
- [Graph Partial Label Learning with Potential Cause Discovering](https://arxiv.org/abs/2403.11449) (arXiv 2024)
- [Appeal: Allow Mislabeled Samples the Chance to be Rectified in Partial Label Learning](https://arxiv.org/abs/2312.11034) (arXiv 2024)
- [Does Label Smoothing Help Deep Partial Label Learning?](https://openreview.net/forum?id=drjjxmi2Ha) (ICML 2024)
- [A General Framework for Learning from Weak Supervision](https://arxiv.org/abs/2402.01922) (ICML 2024)
- [CroSel: Cross Selection of Confident Pseudo Labels for Partial-Label Learning](https://arxiv.org/abs/2303.10365) (CVPR 2024)
- [Disentangled Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/28976) (AAAI 2024)
- [Distilling Reliable Knowledge for Instance-Dependent Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29519) (AAAI 2024)
- [Partial Label Learning with a Partner](https://ojs.aaai.org/index.php/AAAI/article/view/29424) (AAAI 2024)
- [Imprecise Label Learning: A Unified Framework for Learning with Various Imprecise Label Configurations](https://arxiv.org/abs/2305.12715) (NeurIPS 2024)
- [Self-distillation and self-supervision for partial label learning](https://www.sciencedirect.com/science/article/pii/S0031320323007136) (PR 2023)
- [Partial Label Learning with Dissimilarity Propagation guided Candidate Label Shrinkage](https://papers.nips.cc/paper_files/paper/2023/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html) (NeurIPS 2023)
- [Learning From Biased Soft Labels](https://arxiv.org/abs/2302.08155) (NeurIPS 2023)
- [Candidate-aware Selective Disambiguation Based On Normalized Entropy for Instance-dependent Partial-label Learning](https://ieeexplore.ieee.org/document/10376678) (ICCV 2023)
- [Towards Effective Visual Representations for Partial-Label Learning](https://arxiv.org/abs/2305.06080) (CVPR 2023)
- [Complementary Classifier Induced Partial Label Learning](https://arxiv.org/abs/2305.09897) (KDD 2023)
- [Consistent Complementary-Label Learning via Order-Preserving Losses](https://proceedings.mlr.press/v206/liu23g.html) (AISTATS 2023) [:octocat:](https://github.com/yzcao-nkg/OPCLL)
- [Can Label-Specific Features Help Partial-Label Learning?](https://ojs.aaai.org/index.php/AAAI/article/view/25904) (AAAI 2023)
- [Decompositional Generation Process for Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2204.03845) (ICLR 2023)
- [Reduction from Complementary-Label Learning to Probability Estimates](https://arxiv.org/abs/2209.09500) (PAKDD 2023)
- [CMW-Net: Learning a Class-Aware Sample Weighting Mapping for Robust Deep Learning](https://arxiv.org/abs/2202.05613) (TPAMI 2023)
- [Variational Label Enhancement](https://ieeexplore.ieee.org/document/9875104) (TPAMI 2023)
- [GraphDPI: Partial label disambiguation by graph representation learning via mutual information maximization](https://www.sciencedirect.com/science/article/abs/pii/S0031320322006136) (PR 2023)
- [Progressive Purification for Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2206.00830) (ICML 2023) [:octocat:](https://github.com/palm-ml/POP)
- [Meta Objective Guided Disambiguation for Partial Label Learning](https://arxiv.org/abs/2208.12459) (arXiv 2023)
- [Solving Partial Label Learning Problem with Multi-Agent Reinforcement Learning](https://openreview.net/forum?id=BNsuf5g-JRd) (OpenReview 2023)
- [Deep Duplex Learning for Weak Supervision](https://openreview.net/forum?id=SeZ5ONageGl) (OpenReview 2023)
- [Partial Label Learning with Semantic Label Representations](https://dl.acm.org/doi/abs/10.1145/3534678.3539434) (KDD 2022)
- [Revisiting Consistency Regularization for Deep Partial Label Learning](https://proceedings.mlr.press/v162/wu22l.html) (ICML 2022)
- [Partial Label Learning via Label Influence Function](https://proceedings.mlr.press/v162/gong22c.html) (ICML 2022)
- [Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning](https://www.ijcai.org/proceedings/2022/502) (IJCAI 2022)
- [Exploring Binary Classification Hidden within Partial Label Learning](https://www.ijcai.org/proceedings/2022/456) (IJCAI 2022)
- [Deep Graph Matching for Partial Label Learning](https://www.ijcai.org/proceedings/2022/459) (IJCAI 2022)
- [PiCO: Contrastive Label Disambiguation for Partial Label Learning](https://openreview.net/forum?id=EhYjZy6e1gJ) (ICLR 2022) [:octocat:](https://github.com/hbzju/PiCO)
- [Exploiting Class Activation Value for Partial-Label Learning](https://openreview.net/forum?id=qqdXHUGec9h) (ICLR 2022) [:octocat:](https://github.com/Ferenas/CAVL)
- [Biased Complementary-Label Learning Without True Labels](https://ieeexplore.ieee.org/document/9836971) (TNNLS 2022)
- [Learning with Proper Partial Labels](https://arxiv.org/abs/2112.12303) (Neurocomputing 2022)
- [Adaptive Graph Guided Disambiguation for Partial Label Learning](https://ieeexplore.ieee.org/document/9573413) (TPAMI 2022)
- [Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2110.12911) (NeurIPS 2021) [:octocat:](https://github.com/palm-ml/valen)
- [Discriminative Complementary-Label Learning with Weighted Loss](https://proceedings.mlr.press/v139/gao21d.html) (ICML 2021)
- [Top-k Partial Label Machine](https://ieeexplore.ieee.org/document/9447152) (TNNLS 2021)
- [Discriminative Metric Learning for Partial Label Learning](https://ieeexplore.ieee.org/document/9585342) (TNNLS 2021)
- [Generalized Large Margin -NN for Partial Label Learning](https://ieeexplore.ieee.org/document/9529072) (TMM 2021)
- [Progressive Identification of True Labels for Partial-Label Learning](https://arxiv.org/abs/2002.08053) (ICML 2020)
- [Partial Label Learning with Batch Label Correction](https://ojs.aaai.org/index.php/AAAI/article/view/6132) (AAAI 2020)
- [Generative-Discriminative Complementary Learning](https://arxiv.org/abs/1904.01612) (AAAI 2020)
- [Deep Discriminative CNN with Temporal Ensembling for Ambiguously-Labeled Image Classification](https://ojs.aaai.org/index.php/AAAI/article/view/6959) (AAAI 2020)
- [Network Cooperation with Progressive Disambiguation for Partial Label Learning](https://arxiv.org/abs/2002.11919) (ECML-PKDD 2020)
- [Large Margin Partial Label Machine](https://ieeexplore.ieee.org/document/8826247) (TNNLS 2020)
- [A Self-Paced Regularization Framework for Partial-Label Learning](https://ieeexplore.ieee.org/document/9094702) (TYCB 2020)
- [Adaptive Graph Guided Disambiguation for Partial Label Learning](https://dl.acm.org/doi/10.1145/3292500.3330840) (KDD 2019)
- [Partial Label Learning by Semantic Difference Maximization](https://www.ijcai.org/proceedings/2019/318) (IJCAI 2019)
- [Partial Label Learning with Self-Guided Retraining](https://arxiv.org/abs/1902.03045) (AAAI 2019)
- [Partial Label Learning via Label Enhancement](https://ojs.aaai.org/index.php/AAAI/article/view/4497) (AAAI 2019)
- [GM-PLL: Graph Matching based Partial Label Learning](https://arxiv.org/abs/1901.03073) (TKDE 2019)
- [A Conditional Multinomial Mixture Model for Superset Label Learning](https://papers.nips.cc/paper_files/paper/2012/hash/aaebdb8bb6b0e73f6c3c54a0ab0c6415-Abstract.html) (NeurIPS 2012)
- [Weakly-Supervised Contrastive Learning for Imprecise Class Labels](https://arxiv.org/abs/2505.22028) (ICML 2025) [:octocat:](https://github.com/Speechless-10308/WSC)
- [Robust Partial-Label Learning by Leveraging Class Activation Values](https://arxiv.org/abs/2502.11743) (ML 2025)
- [KMT-PLL: K-Means Cross-Attention Transformer for Partial Label Learning](https://ieeexplore.ieee.org/abstract/document/10384739) (TNNLS 2024)
- [Exploiting the Potential Supervision Information of Clean Samples in Partial Label Learning](https://arxiv.org/abs/2505.09354) (arXiv 2025)

<a name="problem-settings-and-extensions" />

### <u>Problem Settings & Extensions</u>

<a name="partial-multi-label-learning" />

### <u>Partial Multi-Label Learning</u>

- [Rethinking Consistent Multi-Label Classification Under Inexact Supervision](https://arxiv.org/abs/2510.04091) (arXiv 2025)
- [Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge](https://arxiv.org/abs/2507.05992) (arXiv 2025)
- [Noise-Resistant Label Reconstruction Feature Selection for Partial Multi-Label Learning](https://arxiv.org/abs/2506.04669) (arXiv 2025)
- [Revisiting Sparsity Constraint Under High-Rank Property in Partial Multi-Label Learning](https://arxiv.org/abs/2505.20938) (arXiv 2025)
- [PositiveCoOp: Rethinking Prompting Strategies for Multi-Label Recognition with Partial Annotations](https://arxiv.org/abs/2409.08381) (WACV 2025)
- [Partial Multilabel Learning Using Noise-Tolerant Broad Learning System With Label Enhancement and Dimensionality Reduction](https://ieeexplore.ieee.org/document/103352285) (TNNLS 2025)
- [Reliable Representation Learning for Incomplete Multi-View Missing Multi-Label Classification](https://arxiv.org/abs/2303.17117) (TPAMI 2025)
- [Partial Multi-View Multi-Label Classification via Semantic Invariance Learning and Prototype Modeling](https://openreview.net/forum?id=5ap1MmUqO6) (ICML 2024)
- [A Deep Model for Partial Multi-Label Image Classification with Curriculum Based Disambiguation](https://arxiv.org/abs/2207.02410) (ML 2024)
- [Combining Supervised Learning and Reinforcement Learning for Multi-Label Classification Tasks with Partial Labels](https://arxiv.org/abs/2406.16293) (arXiv 2024)
- [Free Performance Gain from Mixing Multiple Partially Labeled Samples in Multi-label Image Classification](https://arxiv.org/abs/2405.15860) (arXiv 2024)
- [Deep Double Incomplete Multi-View Multi-Label Learning With Incomplete Labels and Missing Views](https://ieeexplore.ieee.org/document/10086538) (TNNLS 2023)
- [Towards Enabling Binary Decomposition for Partial Multi-Label Learning](https://ieeexplore.ieee.org/document/10168295) (TPAMI 2023)
- [Deep Partial Multi-Label Learning with Graph Disambiguation](https://arxiv.org/abs/2305.05882) (IJCAI 2023)
- [Partial Multi-Label Learning with Probabilistic Graphical Disambiguation](https://papers.nips.cc/paper_files/paper/2023/hash/04e05ba5cbc36044f6499d1edf15247e-Abstract-Conference.html) (NeurIPS 2023)
- [ProPML: Probability Partial Multi-label Learning](https://arxiv.org/abs/2403.07603) (DSAA 2023)
- [Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels](https://arxiv.org/abs/2203.02172) (AAAI 2022)
- [Structured Semantic Transfer for Multi-Label Recognition with Partial Labels](https://arxiv.org/abs/2112.10941) (AAAI 2022)
- [Boosting Multi-Label Image Classification with Complementary Parallel Self-Distillation](https://arxiv.org/abs/2205.10986) (IJCAI 2022)
- [Multi-label Classification with Partial Annotations using Class-aware Selective Loss](https://arxiv.org/abs/2110.10955) (CVPR 2022)
- [PLMCL: Partial-Label Momentum Curriculum Learning for Multi-Label Image Classification](https://arxiv.org/abs/2208.09999) (ECCVw 2022)
- [Prior Knowledge Regularized Self-Representation Model for Partial Multilabel Learning](https://ieeexplore.ieee.org/document/9533180) (TYCB 2021)
- [Global-Local Label Correlation for Partial Multi-Label Learning](https://ieeexplore.ieee.org/document/9343691) (TMM 2021)
- [Progressive Enhancement of Label Distributions for Partial Multilabel Learning](https://ieeexplore.ieee.org/document/9615493) (TNNLS 2021)
- [Partial Multi-Label Learning With Noisy Label Identification](https://ieeexplore.ieee.org/document/9354590) (TPAMI 2021)
- [Partial Multi-Label Learning via Credible Label Elicitation](https://ieeexplore.ieee.org/document/9057438) (TPAMI 2021)
- [Learning from Complementary Labels via Partial-Output Consistency Regularization](https://www.ijcai.org/proceedings/2021/423) (IJCAI 2021)
- [Partial Multi-Label Learning with Meta Disambiguation](https://dl.acm.org/doi/abs/10.1145/3447548.3467259) (KDD 2021)
- [Understanding Partial Multi-Label Learning via Mutual Information](https://proceedings.neurips.cc/paper/2021/hash/217c0e01c1828e7279051f1b6675745d-Abstract.html) (NeurIPS 2021)
- [Multi-View Partial Multi-Label Learning with Graph-Based Disambiguation](https://ojs.aaai.org/index.php/AAAI/article/view/5761) (AAAI 2020)
- [Partial Multi-Label Learning via Multi-Subspace Representation](https://www.ijcai.org/proceedings/2020/362) (IJCAI 2020)
- [Feature-Induced Manifold Disambiguation for Multi-View Partial Multi-label Learning](https://dl.acm.org/doi/10.1145/3394486.3403098) (KDD 2020)
- [Learning a Deep ConvNet for Multi-label Classification with Partial Labels](https://arxiv.org/abs/1902.09720) (CVPR 2019)
- [WPML3CP: Wasserstein Partial Multi-Label Learning with Dual Label Correlation Perspectives](https://www.ijcai.org/proceedings/2025/) (IJCAI 2025)

<a name="noisy-partial-label-learning" />

### <u>Noisy Partial Label Learning</u>

- [Bridging Weakly-Supervised Learning and VLM Distillation: Noisy Partial Label Learning for Efficient Downstream Adaptation](https://arxiv.org/abs/2506.03229) (arXiv 2025)
- [Noise Separation guided Candidate Label Reconstruction for Noisy Partial Label Learning](https://openreview.net/forum?id=TOahfjA3sP) (ICLR 2025) [:octocat:](https://github.com/pruirui/PLRC)
- [Pseudo-labelling meets Label Smoothing for Noisy Partial Label Learning](https://arxiv.org/abs/2402.04835) (arXiv 2024)
- [PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning](https://arxiv.org/abs/2201.08984) (TPAMI 2023)
- [On the Robustness of Average Losses for Partial-Label Learning](https://arxiv.org/abs/2106.06152) (TPAMI 2023)
- [FREDIS: A Fusion Framework of Refinement and Disambiguation for Unreliable Partial Label Learning](https://proceedings.mlr.press/v202/qiao23b.html) (ICML 2023)
- [Unreliable Partial Label Learning with Recursive Separation](https://arxiv.org/abs/2302.09891) (IJCAI 2023)
- [ALIM: Adjusting Label Importance Mechanism for Noisy Partial Label Learning](https://arxiv.org/abs/2301.12077) (NeurIPS 2023)
- [Mutual Partial Label Learning with Competitive Label Noise](https://openreview.net/forum?id=EUrxG8IBCrC) (ICLR 2023)
- [IRNet: Iterative Refinement Network for Noisy Partial Label Learning](https://arxiv.org/abs/2211.04774) (arXiv 2023)
- [Robust Representation Learning for Unreliable Partial Label Learning](https://arxiv.org/abs/2308.16718) (arXiv 2023)
- [Learning with Noisy Partial Labels by Simultaneously Leveraging Global and Local Consistencies](https://dl.acm.org/doi/10.1145/3340531.3411885) (CIKM 2020)

<a name="semi-supervised-partial-label-learning" />

### <u>Semi-Supervised Partial Label Learning</u>

- [Semi-Supervised Partial Label Learning via Label Confidence Recovery](https://palm.seu.edu.cn/zhangml/files/PRJ%2726b.pdf) (PR 2026)
- [Partial Label Learning with Semi-Supervised Clustering Disambiguation](https://palm.seu.edu.cn/zhangml/files/PRJ%2726.pdf) (PR 2026)
- [Learning with Partial-Label and Unlabeled Data: A Uniform Treatment for Supervision Redundancy and Insufficiency](https://proceedings.mlr.press/v235/liu24ar.html) (ICML 2024)
- [FairMatch: Promoting Partial Label Learning by Unlabeled Samples](https://dl.acm.org/doi/abs/10.1145/3637528.3671685) (KDD 2024)
- [Learning with Partial Labels from Semi-supervised Perspective](https://arxiv.org/abs/2211.13655) (AAAI 2023)
- [Distributed Semisupervised Partial Label Learning Over Networks](https://ieeexplore.ieee.org/document/9699063) (AI 2022)
- [Exploiting Unlabeled Data via Partial Label Assignment for Multi-Class Semi-Supervised Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17310) (AAAI 2021)
- [Partial Label Learning with Unlabeled Data](https://www.ijcai.org/proceedings/2019/521) (IJCAI 2019)
- [Semi-Supervised Partial Label Learning via Confidence-Rated Margin Maximization](https://proceedings.neurips.cc/paper/2020/hash/4dea382d82666332fb564f2e711cbc71-Abstract.html) (NeurIPS 2020)

<a name="multi-instance-learning" />

### <u>Multi-Instance Partial Label Learning</u>

- [Fast Multi-Instance Partial-Label Learning](https://palm.seu.edu.cn/zhangml/files/AAAI%2725c.pdf) (AAAI 2025)
- [Calibratable Disambiguation Loss for Multi-Instance Partial-Label Learning](https://arxiv.org/abs/2512.17788) (arXiv 2025)
- [On Characterizing and Mitigating Imbalances in Multi-Instance Partial Label Learning](https://openreview.net/forum?id=oZdaEiDBpF) (ICLR 2025)
- [Multi-Instance Partial-Label Learning with Margin Adjustment](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2e5a74735e5ada01d85afbb49caad36f-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/tangw-seu/MIPLMA)
- [Exploiting Conjugate Label Information for Multi-Instance Partial-Label Learning](https://arxiv.org/abs/2408.14369) (arXiv 2024)
- [Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning](https://arxiv.org/abs/2305.16912) (NeurIPS 2023)
- [Multi-Instance Partial-Label Learning: Towards Exploiting Dual Inexact Supervision](https://arxiv.org/abs/2212.08997) (SCIS 2023)

<a name="class-imbalance-problem" />

### <u>Imbalanced Partial Label Learning</u>

- [Long-Tailed Partial Label Learning by Head Classifier and Tail Classifier Cooperation](https://ojs.aaai.org/index.php/AAAI/article/view/29182) (AAAI 2024)
- [Pseudo Labels Regularization for Imbalanced Partial-Label Learning](https://arxiv.org/abs/2303.03946) (ICASSP 2024)
- [Learning in Imperfect Environment: Multi-Label Classification with Long-Tailed Distribution and Partial Labels](https://arxiv.org/abs/2304.10539) (ICCV 2023)
- [Class-Imbalanced Complementary-Label Learning via Weighted Loss](https://arxiv.org/abs/2209.14189) (NN 2023)
- [Long-Tailed Partial Label Learning via Dynamic Rebalancing](https://arxiv.org/abs/2302.05080) (ICLR 2023)
- [SoLar: Sinkhorn Label Refinery for Imbalanced Partial-Label Learning](https://arxiv.org/abs/2209.10365) (NeurIPS 2022)
- [A Partial Label Metric Learning Algorithm for Class Imbalanced Data](https://proceedings.mlr.press/v157/liu21f.html) (ACML 2021)
- [Towards Mitigating the Class-Imbalance Problem for Partial Label Learning](https://dl.acm.org/doi/10.1145/3219819.3220008) (KDD 2018) [:octocat:](https://github.com/seu71wj/CIMAP)
- [GBRIP: Granular Ball Representation for Imbalanced Partial Label Learning](https://palm.seu.edu.cn/zhangml/) (PR 2024)

<a name="ood-pll" />

### Out-of-distribution Partial Label Learning

- [Out-of-distribution Partial Label Learning](https://arxiv.org/abs/2403.06681) (arXiv 2024)

<a name="fed-pll" />

### Federated Partial Label Learning

- [Federated Partial Label Learning with Local-Adaptive Augmentation and Regularization](https://ojs.aaai.org/index.php/AAAI/article/view/29562) (AAAI 2024)

<a name="partial-label-regression" />

### <u>Partial Label Regression</u>

- [Partial-Label Learning with a Reject Option](https://arxiv.org/abs/2402.00592) (TMLR 2024)
- [Partial-Label Regression](https://arxiv.org/abs/2306.08968) (AAAI 2023)

<a name="dimensionality-reduction" />

### <u>Dimensionality Reduction</u>

- [Dimensionality Reduction for Partial Label Learning: A Unified and Adaptive Approach](https://www.computer.org/csdl/journal/tk/2024/08/10440495/1UGSdp8q3Wo) (TKDE 2024)
- [Disambiguation Enabled Linear Discriminant Analysis for Partial Label Dimensionality Reduction](https://dl.acm.org/doi/abs/10.1145/3494565) (TKDD 2022)
- [Submodular Feature Selection for Partial Label Learning](https://dl.acm.org/doi/abs/10.1145/3534678.3539292) (KDD 2022)
- [Partial Label Dimensionality Reduction via Confidence-Based Dependence Maximization](https://dl.acm.org/doi/abs/10.1145/3447548.3467313) (KDD 2021)
- [Wrapped Partial Label Dimensionality Reduction via Dependence Maximization](https://www.ijcai.org/proceedings/2025/0769) (IJCAI 2025)

<a name="learning-paradigms-and-strategies" />

### <u>Learning Paradigms & Strategies</u>

<a name="multi-complementary-label-learning" />

### <u>Multi-Complementary Label Learning</u>

- [Data Augmentation with Regularization for Multi-Labeled Complementary Label Learning](https://palm.seu.edu.cn/zhangml/files/TPAMI%2725c.pdf) (TPAMI 2026)
- [Multi-Label Learning with Multiple Complementary Labels](https://palm.seu.edu.cn/zhangml/files/TPAMI%2725.pdf) (TPAMI 2025)
- [Complementary to Multiple Labels: A Correlation-Aware Correction Approach](https://palm.seu.edu.cn/zhangml/files/TPAMI%2724b.pdf) (TPAMI 2024)
- [Multi-Complementary and Unlabeled Learning for Arbitrary Losses and Models](https://arxiv.org/abs/2001.04243) (PR 2022)
- [Learning with Multiple Complementary Labels](https://arxiv.org/abs/1912.12927) (ICML 2020)
- [Complementary Label Learning with Positive Label Guessing and Negative Label Enhancement](https://openreview.net/forum?id=LPRxGZ7Oax) (ICLR 2025)

<a name="multi-view-learning" />

### <u>Multi-View Learning</u>

- [Deep Partial Multi-View Learning](https://ieeexplore.ieee.org/document/9258396) (TPAMI 2022)

<a name="adversarial-training" />

### <u>Adversarial Training</u>

- [Adversary-Aware Partial label learning with Label distillation](https://arxiv.org/abs/2304.00498) (arXiv 2023)
- [Adversarial Training with Complementary Labels: On the Benefit of Gradually Informative Attacks](https://openreview.net/forum?id=s7SukMH7ie9) (NeurIPS 2022)
- [Detecting the Fake Candidate Instances: Ambiguous Label Learning with Generative Adversarial Networks](https://dl.acm.org/doi/abs/10.1145/3459637.3482251) (CIKM 2021)
- [Adversarial Partial Multi-Label Learning](https://arxiv.org/abs/1909.06717) (AAAI 2021)

<a name="negative-learning" />

### <u>Negative Learning</u>

+ To be continued.

<a name="incremental-learning" />

### <u>Incremental Learning</u>

- [An Unbiased Risk Estimator for Partial Label Learning with Augmented Classes](https://arxiv.org/abs/2409.19600) (arXiv 2024)
- [Partial label learning with emerging new labels](https://link.springer.com/article/10.1007/s10994-022-06244-2) (ML 2022)
- [Complementary Labels Learning with Augmented Classes](https://arxiv.org/abs/2211.10701) (arXiv 2022)

<a name="online-learning" />

### <u>Online Learning</u>

- [Online Partial Label Learning](https://dl.acm.org/doi/abs/10.1007/978-3-030-67661-2_27) (ECML-PKDD 2020)

<a name="conformal-prediction" />

### <u>Conformal Prediction</u>

- [Conformal Prediction with Partially Labeled Data](https://arxiv.org/abs/2306.01191) (SCPPA 2023)
- [Partial-Label Learning with Conformal Candidate Cleaning](https://arxiv.org/abs/2502.07661) (UAI 2025)

<a name="few-shot-learning" />

### <u>Few-Shot Learning</u>

- [Few-Shot Partial-Label Learning](https://arxiv.org/abs/2106.00984) (IJCAI 2021)

<a name="open-set-problem" />

### <u>Open-set Problem</u>

- [Partial-label Learning with Mixed Closed-set and Open-set Out-of-candidate Examples](https://arxiv.org/abs/2307.00553) (KDD 2023)

<a name="data-augmentation" />

### <u>Data Augmentation</u>

- [Enhancing Label Sharing Efficiency in Complementary-Label Learning with Label Augmentation](https://arxiv.org/abs/2305.08344) (arXiv 2023)
- [Partial Label Learning with Discrimination Augmentation](https://dl.acm.org/doi/abs/10.1145/3534678.3539363) (KDD 2022) [:octocat:](https://github.com/wwangwitsel/PLDA)

<a name="multi-dimensional" />

### <u>Multi-Dimensional</u>

- [Learning From Multi-Dimensional Partial Labels](https://www.ijcai.org/proceedings/2020/407) (IJCAI 2020)

<a name="domain-adaptation" />

### <u>Domain Adaptation</u>

- [Partial Label Causal Representation Learning for Instance-Dependent Supervision and Domain Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/35437) (AAAI 2025)
- [Partial Label Unsupervised Domain Adaptation with Class-Prototype Alignment](https://openreview.net/forum?id=jpq0qHggw3t) (ICLR 2023)



<a name="applications" />

## <u>Applications</u>

<a name="audio" />

### <u>Audio</u>

- [Semi-Supervised Audio Classification with Partially Labeled Data](https://arxiv.org/abs/2111.12761) (arXiv 2021)

<a name="text"/>

### <u>Text</u>

- [Complementary Auxiliary Classifiers for Label-Conditional Text Generation](https://ojs.aaai.org/index.php/AAAI/article/view/6346) (AAAI 2020)

<a name="sequence" />

### <u>Sequence</u>

- [Star Temporal Classification: Sequence Classification with Partially Labeled Data](https://arxiv.org/abs/2201.12208) (arXiv 2022)

<a name="recognition"/>

### <u>Recognition</u>

- [W2P: Switching from Weak Supervision to Partial Supervision for Semantic Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/28531) (AAAI 2024)
- [Partial Label Learning with Focal Loss for Sea Ice Classification Based on Ice Charts](https://arxiv.org/abs/2406.03645) (AEORS 2023)
- [Partial Label Learning for Emotion Recognition from EEG](https://arxiv.org/abs/2302.13170) (arXiv 2023)
- [A Confidence-based Partial Label Learning Model for Crowd-Annotated Named Entity Recognition](https://arxiv.org/abs/2305.12485) (ACL 2023) [:octocat:](https://github.com/LemXiong/CPLL)
- [Webly-Supervised Fine-Grained Recognition with Partial Label Learning](https://www.ijcai.org/proceedings/2022/209) (IJCAI 2022) [:octocat:](https://github.com/msfuxian/WSFGPLL)

<a name="object-localization"/>

### <u>Object Localization</u>

- [Adversarial Complementary Learning for Weakly Supervised Object Localization](https://arxiv.org/abs/1804.06962) (CVPR 2018) [:octocat:](https://github.com/halbielee/ACoL_pytorch)
- [Learning to Detect Instance-level Salient Objects Using Complementary Image Labels](https://arxiv.org/abs/2111.10137) (arXiv 2021)

<a name="map-reconstruction"/>

### <u>Map Reconstruction</u>

- [Deep Learning with Partially Labeled Data for Radio Map Reconstruction](https://arxiv.org/abs/2306.05294) (arXiv 2023)

<a name="semi-supervised-learning"/>

### <u>Semi-supervised Learning</u>

- [Controller-Guided Partial Label Consistency Regularization with Unlabeled Data](https://arxiv.org/abs/2210.11194) (arXiv 2024)
- [Semi-supervised Contrastive Learning Using Partial Label Information](https://arxiv.org/abs/2003.07921) (arXiv 2024)
- [Boosting Semi-Supervised Learning with Contrastive Complementary Labeling](https://arxiv.org/abs/2212.06643) (NN 2023)

<a name="active-learning"/>

### <u>Active Learning</u>

- [Exploiting counter-examples for active learning with partial labels](https://link.springer.com/article/10.1007/s10994-023-06485-9) (ML 2023) [:octocat:](https://github.com/Ferenas/APLL)
- [Active Learning with Partial Labels](https://openreview.net/forum?id=1VuBdlNBuR) (OpenReview 2023)

<a name="noisy-label-learning"/>

### <u>Noisy Label Learning</u>

- [Adaptive Integration of Partial Label Learning and Negative Learning for Enhanced Noisy Label Learning](https://arxiv.org/abs/2312.09505) (AAAI 2024)
- [Mitigating Label Noise through Data Ambiguation](https://ojs.aaai.org/index.php/AAAI/article/view/29286) (AAAI 2024) [:octocat:](https://github.com/julilien/mitigatinglabelnoisedataambiguation)
- [Partial Label Supervision for Agnostic Generative Noisy Label Learning](https://arxiv.org/abs/2308.01184) (arXiv 2024)
- [Learning from Noisy Labels with Complementary Loss Functions](https://ojs.aaai.org/index.php/AAAI/article/view/17213) (AAAI 2021)
- [NLNL: Negative Learning for Noisy Labels](https://arxiv.org/abs/1908.07387) (ICCV 2019)

<a name="TTA"/>

### <u>Test-Time Adaptation</u>

- [Rethinking Precision of Pseudo Label: Test-Time Adaptation via Complementary Learning](https://arxiv.org/abs/2301.06013) (arXiv 2023)

<a name="weak-supervision-evaluation"/>

### <u>Weak Supervision Evaluation</u>

- [Weak Supervision Performance Evaluation via Partial Identification](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f4c6bec746b0aeca8c2cd15096f1ad1f-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/felipemaiapolo/wsbounds)




## Dataset

### Tabular Dataset:

**Notice:** The following partial label learning data sets were collected and pre-processed by Prof. [Min-Ling Zhang](https://palm.seu.edu.cn/zhangml/), with courtesy and proprietary to the authors of referred literatures on them. The pre-processed data sets can be used at your own risk and for academic purpose only. More information can be found in [here](http://palm.seu.edu.cn/zhangml/).

Dataset for partial label learning:

| [FG-NET](https://palm.seu.edu.cn/zhangml/files/FG-NET.rar) | [Lost](https://palm.seu.edu.cn/zhangml/files/lost.rar) | [MSRCv2](https://palm.seu.edu.cn/zhangml/files/MSRCv2.rar) | [BirdSong](https://palm.seu.edu.cn/zhangml/files/BirdSong.rar) | [Soccer Player](https://palm.seu.edu.cn/zhangml/files/SoccerPlayer.rar) | [Yahoo! News](https://palm.seu.edu.cn/zhangml/files/Yahoo!News.rar) | [Mirflickr](https://palm.seu.edu.cn/zhangml/files/Mirflickr.zip) |
| ---------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

Dataset for partial multi-label learning:

| [Music_emotion](https://palm.seu.edu.cn/zhangml/files/pml_music_emotion.zip) | [Music_style](https://palm.seu.edu.cn/zhangml/files/pml_music_style.zip) | [Mirflickr](https://palm.seu.edu.cn/zhangml/files/pml_mirflickr.zip) | [YeastBP](https://palm.seu.edu.cn/zhangml/files/pml_YeastBP.zip) | [YeastCC](https://palm.seu.edu.cn/zhangml/files/pml_YeastCC.zip) | [YeastMF](https://palm.seu.edu.cn/zhangml/files/pml_YeastMF.zip) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

Data sets for multi-instance partial-label learning:

| [MNIST](https://palm.seu.edu.cn/zhangml/files/MNIST_MIPL.zip) | [FMNIST](https://palm.seu.edu.cn/zhangml/files/FMNIST_MIPL.zip) | [Newsgroups](https://palm.seu.edu.cn/zhangml/files/Newsgroups_MIPL.zip) | [Birdsong](https://palm.seu.edu.cn/zhangml/files/Birdsong_MIPL.zip) | [SIVAL](https://palm.seu.edu.cn/zhangml/files/SIVAL_MIPL.zip) | [CRC-Row](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-Row.zip) | [CRC-SBN](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-SBN.zip) | [CRC-KMeansSeg](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-KMeansSeg.zip) | [CRC-SIFT](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-SIFT.zip) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

### Image Dataset:

- Dataset for complementary-label learning: [CLImage dataset](https://github.com/ntucllab/CLImage_Dataset)
- Dataset for partial-label learning: [PLENCH dataset](https://github.com/wwangwitsel/PLENCH)



## Leaderboard

To be continued.



## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=wu-dd/Advances-in-Partial-and-Complementary-Label-Learning&type=Date)](https://star-history.com/#wu-dd/Advances-in-Partial-and-Complementary-Label-Learning&Date)

## Citing Advances in Partial/Complementary Label Learning
If you find this project useful for your research, please use the following BibTeX entry.
```
@misc{Wu2022advances,
  author={Dong-Dong Wu},
  title={Advances in Partial/Complementary Label Learning },
  howpublished={\url{wu-dd/Advances-in-Partial-and-Complementary-Label-Learning}},
  year={2022}
}
```

## Acknowledgments
