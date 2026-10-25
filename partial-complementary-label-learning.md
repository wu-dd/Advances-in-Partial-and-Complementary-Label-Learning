# Advances in Partial/Complementary Label Learning 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/> <img src="https://img.shields.io/badge/Number%20of%20Items-332-FF6F00" alt="PaperNum"/>

**Advances in Partial/Complementary Label Learning** provides the most advanced and detailed information on partial/complementary label learning field.

**Partial/complementary label learning** is an emerging framework in weakly supervised machine learning with broad application prospects. It handles the case in which each training example corresponds to a candidate label set and only one label concealed in the set is the ground-truth label.

This list is part of [Advances in Learning with Imperfect Information](./README.md), curated and maintained by [Dong-Dong Wu](https://wu-dd.github.io/). You can also [browse, search and filter it online](https://wu-dd.github.io/Advances-in-Learning-with-Imperfect-Information/#pll). Suggestions are welcome by email.

+ :octocat: Code (174 papers have official code)

## Latest Updates
+ [2026/10/24] Checked every entry: fixed broken/wrong paper links and venues (many arXiv entries are now published), added official code links (now 174 papers with code), removed off-topic and duplicate entries, reorganized the categories (new sections for instance-dependent PLL, complementary-label learning, foundation models, multi-label learning with partial annotations, graph and cross-modal retrieval), and added 130+ papers from top venues in 2024-2026.
+ [2026/03/16] Conducted another venue-by-venue and journal-by-journal semantic search (Google Scholar / Semantic Scholar / arXiv and venue pages), and added 20+ missing papers across NeurIPS/AAAI/TPAMI/PR/arXiv tracks (e.g., ComRank, What Makes PLL Effective, CausalPLL, Fast-MIPL, PR 2026 semi-supervised PLL papers, and several new arXiv 2025 PLL/partial-multi-label works).
+ [2025/09/22] Update some papers published on conferences. If you find some errors, please feel free to contact me!
+ [2024/12/08] Update some code links of papers.
+ [2024/09/13] A major overhaul of the original github repository. 

<!-- papers:start -->
## Contents
- [Foundations & Surveys](#foundations-surveys)
  - [Early Work](#early-work)
  - [Survey](#survey)
  - [Benchmark & Datasets](#benchmark)
  - [Theory & Risk-Consistent Methods](#theory-based)
  - [Understanding & Analysis](#understanding)
- [Algorithms](#algorithms)
  - [Partial-Label Learning Methods](#better-optimization)
  - [Instance-Dependent Partial-Label Learning](#instance-dependent-pll)
  - [Learning with Foundation / Vision-Language Models](#foundation-models)
  - [Complementary-Label Learning Methods](#complementary-label-learning)
  - [Multiple / Multi-Label Complementary-Label Learning](#multi-complementary-label-learning)
  - [Dimensionality Reduction & Feature Selection](#dimensionality-reduction)
  - [Data Augmentation](#data-augmentation)
- [Problem Settings & Extensions](#problem-settings-extensions)
  - [Partial Multi-Label Learning](#partial-multi-label-learning)
  - [Multi-Label Learning with Partial Annotations](#multi-label-partial-annotations)
  - [Noisy Partial-Label Learning](#noisy-partial-label-learning)
  - [Semi-Supervised Partial-Label Learning](#semi-supervised-partial-label-learning)
  - [Multi-Instance Partial-Label Learning](#multi-instance-learning)
  - [Imbalanced Partial-Label Learning](#class-imbalance-problem)
  - [Open-Set & Out-of-Distribution Partial-Label Learning](#open-set-problem)
  - [Incremental Learning & Augmented Classes](#incremental-learning)
  - [Online Learning](#online-learning)
  - [Few-Shot Learning](#few-shot-learning)
  - [Domain Adaptation & Generalization](#domain-adaptation)
  - [Federated Partial-Label Learning](#fed-pll)
  - [Partial-Label Regression](#partial-label-regression)
  - [Reject Option & Conformal Prediction](#conformal-prediction)
  - [Multi-Dimensional Partial-Label Learning](#multi-dimensional)
  - [Multi-View Partial-Label Learning](#multi-view-learning)
  - [Adversarial Training & Robustness](#adversarial-training)
- [Applications](#applications)
  - [Large Language Models](#llm)
  - [Natural Language Processing](#text)
  - [Sequence](#sequence)
  - [Computer Vision](#recognition)
  - [Graph Learning](#graph)
  - [Cross-Modal Retrieval & Hashing](#retrieval)
  - [Recommender Systems](#recommendation)
  - [Semi-Supervised Learning](#semi-supervised-learning)
  - [Active Learning](#active-learning)
  - [Noisy-Label Learning](#noisy-label-learning)
  - [Test-Time Adaptation](#TTA)
  - [Weak Supervision Evaluation](#weak-supervision-evaluation)

<a name="foundations-surveys" />

## <u>Foundations & Surveys</u>

<a name="early-work" />

### <u>Early Work</u>

- [Learning from Complementary Labels](https://arxiv.org/abs/1705.07541) (NeurIPS 2017) [:octocat:](https://github.com/takashiishida/comp)
- [A Conditional Multinomial Mixture Model for Superset Label Learning](https://papers.nips.cc/paper_files/paper/2012/hash/aaebdb8bb6b0e73f6c3c54a0ab0c6415-Abstract.html) (NeurIPS 2012)
- [Learning from Partial Labels](https://jmlr.org/papers/v12/cour11a.html) (JMLR 2011)

<a name="survey" />

### <u>Survey</u>

- [Complementary-Label Learning: A Survey of Methods, an Open-Source Toolkit, and Fair Benchmarking Results](https://doi.org/10.1145/3847508) (CSUR 2026) [:octocat:](https://github.com/ntucllab/libcll)
- [Partial label learning: Taxonomy, analysis and outlook](https://www.sciencedirect.com/science/article/abs/pii/S0893608023000825) (NN 2023)

<a name="benchmark" />

### <u>Benchmark & Datasets</u>

- [CLImage: Human-Annotated Datasets for Complementary-Label Learning](https://arxiv.org/abs/2305.08295) (TMLR 2025) [:octocat:](https://github.com/ntucllab/CLImage_Dataset)
- [Realistic Evaluation of Deep Partial-Label Learning Algorithms](https://arxiv.org/abs/2502.10184) (ICLR 2025)

<a name="theory-based" />

### <u>Theory & Risk-Consistent Methods</u>

- [Learnability with Partial Labels and Adaptive Nearest Neighbors](https://proceedings.mlr.press/v300/errandonea26a.html) (AISTATS 2026) [:octocat:](https://github.com/MachineLearningBCAM/Adaptive-nearest-neighbours-for-partial-labels)
- [A Unified View of Forward and Backward Losses for Learning from Weak Labels](https://doi.org/10.1007/s10994-025-06841-x) (ML 2025)
- [Learning with Complementary Labels Revisited: The Selected-Completely-at-Random Setting Is More Practical](https://arxiv.org/abs/2311.15502) (ICML 2024) [:octocat:](https://github.com/wwangwitsel/SCARCE)
- [Towards Unbiased Exploration in Partial Label Learning](https://arxiv.org/abs/2307.00465) (JMLR 2024) [:octocat:](https://github.com/agapiR/partial-label-learning)
- [Consistent Complementary-Label Learning via Order-Preserving Losses](https://proceedings.mlr.press/v206/liu23g.html) (AISTATS 2023) [:octocat:](https://github.com/yzcao-nkg/OPCLL)
- [Reduction from Complementary-Label Learning to Probability Estimates](https://arxiv.org/abs/2209.09500) (PAKDD 2023) [:octocat:](https://github.com/ntucllab/libcll)
- [Leveraged Weighted Loss for Partial Label Learning](https://arxiv.org/abs/2106.05731) (ICML 2021) [:octocat:](https://github.com/hongwei-wen/LW-loss-for-partial-label)
- [Provably Consistent Partial-Label Learning](https://arxiv.org/abs/2007.08929) (NeurIPS 2020) [:octocat:](https://lfeng1995.github.io/Code/RCCC.zip)
- [Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels](https://arxiv.org/abs/2007.02235) (ICML 2020)
- [Bridging Ordinary-Label Learning and Complementary-Label Learning](https://proceedings.mlr.press/v129/katsura20a.html) (ACML 2020) [:octocat:](https://github.com/YasuhiroKatsura/ord-comp)
- [Complementary-Label Learning for Arbitrary Losses and Models](https://arxiv.org/abs/1810.04327) (ICML 2019) [:octocat:](https://github.com/takashiishida/comp)
- [Learning with Biased Complementary Labels](https://arxiv.org/abs/1711.09535) (ECCV 2018)

<a name="understanding" />

### <u>Understanding & Analysis</u>

- [Rethinking Self-Distillation: Label Averaging and Enhanced Soft Label Refinement with Partial Labels](https://arxiv.org/abs/2402.10482) (ICLR 2025)
- [Candidate Label Set Pruning: A Data-centric Perspective for Deep Partial-label Learning](https://openreview.net/forum?id=Fk5IzauJ7F) (ICLR 2024) [:octocat:](https://github.com/shuohe1995/CLSP)
- [What Makes Partial-Label Learning Algorithms Effective?](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a2f1a33355610e271b8a27f09dac83d2-Abstract-Conference.html) (NeurIPS 2024)
- [Does Label Smoothing Help Deep Partial Label Learning?](https://openreview.net/forum?id=drjjxmi2Ha) (ICML 2024) [:octocat:](https://github.com/kalpiree/LS-PLL)
- [A Unifying Probabilistic Framework for Partially Labeled Data Learning](https://ieeexplore.ieee.org/document/9983986) (TPAMI 2023)
- [On the Power of Deep but Naive Partial Label Learning](https://arxiv.org/abs/2010.11600) (ICASSP 2021) [:octocat:](https://github.com/mikigom/DNPL-PyTorch)

<a name="algorithms" />

## <u>Algorithms</u>

<a name="better-optimization" />

### <u>Partial-Label Learning Methods</u>

- [Diffusion Disambiguation Models for Partial Label Learning](https://arxiv.org/abs/2507.00411) (TCSVT 2026) [:octocat:](https://github.com/fanjinfucool/DDMP)
- [Exploiting the Potential Supervision Information of Clean Samples in Partial Label Learning](https://arxiv.org/abs/2505.09354) (PR 2026)
- [Amortized Variational Inference for Partial-Label Learning: A Probabilistic Approach to Label Disambiguation](https://proceedings.mlr.press/v306/fuchs26a.html) (ICML 2026) [:octocat:](https://github.com/mathefuchs/vi-pll)
- [Evidential Deep Partial Label Learning to Quantify Disambiguation Uncertainty](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_Evidential_Deep_Partial_Label_Learning_to_Quantify_Disambiguation_Uncertainty_CVPR_2026_paper.html) (CVPR 2026)
- [Semantic-Aware Feature Enhancement for Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/39616) (AAAI 2026)
- [Mutual Information Guided Reinforcement Learning for Ambiguous Label Disambiguation](https://www.ijcai.org/proceedings/2026/474) (IJCAI 2026)
- [Positive-Negative Complementarity Disambiguation with Semantic-Visual Calibration for Partial Label Learning](https://doi.org/10.1016/j.patcog.2026.115091) (PR 2026)
- [Weakly-Supervised Contrastive Learning for Imprecise Class Labels](https://arxiv.org/abs/2505.22028) (ICML 2025) [:octocat:](https://github.com/Speechless-10308/WSC)
- [Robust Partial-Label Learning by Leveraging Class Activation Values](https://arxiv.org/abs/2502.11743) (ML 2025) [:octocat:](https://github.com/mathefuchs/robust-pll)
- [AsyCo: An Asymmetric Dual-task Co-training Model for Partial-label Learning](https://doi.org/10.1007/s11432-023-4068-1) (SCIS 2025) [:octocat:](https://github.com/libeibeics/AsyCo)
- [Dealing with Partial Labels by Knowledge Distillation](https://doi.org/10.1016/j.patcog.2024.110965) (PR 2025)
- [Partial Label Clustering](https://www.ijcai.org/proceedings/2025/743) (IJCAI 2025) [:octocat:](https://github.com/xyt-ml/PLC)
- [Revising the Problem of Partial Labels from the Perspective of CNNs' Robustness](https://arxiv.org/abs/2407.17630) (SERA 2024)
- [Appeal: Allow Mislabeled Samples the Chance to be Rectified in Partial Label Learning](https://arxiv.org/abs/2312.11034) (arXiv 2024)
- [A General Framework for Learning from Weak Supervision](https://arxiv.org/abs/2402.01922) (ICML 2024) [:octocat:](https://github.com/Hhhhhhao/General-Framework-Weak-Supervision)
- [CroSel: Cross Selection of Confident Pseudo Labels for Partial-Label Learning](https://arxiv.org/abs/2303.10365) (CVPR 2024)
- [Disentangled Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/28976) (AAAI 2024)
- [Partial Label Learning with a Partner](https://ojs.aaai.org/index.php/AAAI/article/view/29424) (AAAI 2024)
- [Imprecise Label Learning: A Unified Framework for Learning with Various Imprecise Label Configurations](https://arxiv.org/abs/2305.12715) (NeurIPS 2024) [:octocat:](https://github.com/Hhhhhhao/General-Framework-Weak-Supervision)
- [KMT-PLL: K-Means Cross-Attention Transformer for Partial Label Learning](https://ieeexplore.ieee.org/abstract/document/10384739) (TNNLS 2024)
- [Self-distillation and self-supervision for partial label learning](https://www.sciencedirect.com/science/article/pii/S0031320323007136) (PR 2023)
- [Partial Label Learning with Dissimilarity Propagation guided Candidate Label Shrinkage](https://papers.nips.cc/paper_files/paper/2023/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html) (NeurIPS 2023) [:octocat:](https://github.com/Yangfc-ML/DPCLS)
- [Learning From Biased Soft Labels](https://arxiv.org/abs/2302.08155) (NeurIPS 2023)
- [Towards Effective Visual Representations for Partial-Label Learning](https://arxiv.org/abs/2305.06080) (CVPR 2023) [:octocat:](https://github.com/eddiexia98/PaPi)
- [Complementary Classifier Induced Partial Label Learning](https://arxiv.org/abs/2305.09897) (KDD 2023) [:octocat:](https://github.com/Chongjie-Si/PL-CL)
- [Can Label-Specific Features Help Partial-Label Learning?](https://ojs.aaai.org/index.php/AAAI/article/view/25904) (AAAI 2023) [:octocat:](https://github.com/meteoseeker/UCL)
- [CMW-Net: Learning a Class-Aware Sample Weighting Mapping for Robust Deep Learning](https://arxiv.org/abs/2202.05613) (TPAMI 2023) [:octocat:](https://github.com/xjtushujun/CMW-Net)
- [Variational Label Enhancement](https://ieeexplore.ieee.org/document/9875104) (TPAMI 2023) [:octocat:](https://github.com/palm-ml/LEVI)
- [GraphDPI: Partial label disambiguation by graph representation learning via mutual information maximization](https://www.sciencedirect.com/science/article/abs/pii/S0031320322006136) (PR 2023)
- [Meta Objective Guided Disambiguation for Partial Label Learning](https://arxiv.org/abs/2208.12459) (arXiv 2023)
- [Learning with Proper Partial Labels](https://arxiv.org/abs/2112.12303) (Neural Computation 2023)
- [Partial Label Learning with Semantic Label Representations](https://dl.acm.org/doi/abs/10.1145/3534678.3539434) (KDD 2022) [:octocat:](https://github.com/shuohe1995/ParSE)
- [Revisiting Consistency Regularization for Deep Partial Label Learning](https://proceedings.mlr.press/v162/wu22l.html) (ICML 2022) [:octocat:](https://palm.seu.edu.cn/zhangml/files/ICML%2722a.rar)
- [Partial Label Learning via Label Influence Function](https://proceedings.mlr.press/v162/gong22c.html) (ICML 2022)
- [Exploring Binary Classification Hidden within Partial Label Learning](https://www.ijcai.org/proceedings/2022/456) (IJCAI 2022) [:octocat:](https://github.com/RUC-DWBI-ML/PLL-BC)
- [Deep Graph Matching for Partial Label Learning](https://www.ijcai.org/proceedings/2022/459) (IJCAI 2022)
- [PiCO: Contrastive Label Disambiguation for Partial Label Learning](https://openreview.net/forum?id=EhYjZy6e1gJ) (ICLR 2022) [:octocat:](https://github.com/hbzju/PiCO)
- [Exploiting Class Activation Value for Partial-Label Learning](https://openreview.net/forum?id=qqdXHUGec9h) (ICLR 2022) [:octocat:](https://github.com/SII-Ferenas/CAVL)
- [Adaptive Graph Guided Disambiguation for Partial Label Learning](https://ieeexplore.ieee.org/document/9573413) (TPAMI 2022) [:octocat:](https://github.com/dengbaowang/PL-AGGD)
- [A Generative Model for Partial Label Learning](https://ieeexplore.ieee.org/abstract/document/9428103) (ICME 2021)
- [Top-k Partial Label Machine](https://ieeexplore.ieee.org/document/9447152) (TNNLS 2021)
- [Discriminative Metric Learning for Partial Label Learning](https://ieeexplore.ieee.org/document/9585342) (TNNLS 2021)
- [Generalized Large Margin kNN for Partial Label Learning](https://ieeexplore.ieee.org/document/9529072) (TMM 2021)
- [Progressive Identification of True Labels for Partial-Label Learning](https://arxiv.org/abs/2002.08053) (ICML 2020) [:octocat:](https://github.com/lvjiaqi77/PRODEN)
- [Partial Label Learning with Batch Label Correction](https://ojs.aaai.org/index.php/AAAI/article/view/6132) (AAAI 2020)
- [Deep Discriminative CNN with Temporal Ensembling for Ambiguously-Labeled Image Classification](https://ojs.aaai.org/index.php/AAAI/article/view/6959) (AAAI 2020)
- [Network Cooperation with Progressive Disambiguation for Partial Label Learning](https://arxiv.org/abs/2002.11919) (ECML-PKDD 2020)
- [Large Margin Partial Label Machine](https://ieeexplore.ieee.org/document/8826247) (TNNLS 2020)
- [A Self-Paced Regularization Framework for Partial-Label Learning](https://ieeexplore.ieee.org/document/9094702) (TCYB 2020)
- [Adaptive Graph Guided Disambiguation for Partial Label Learning](https://dl.acm.org/doi/10.1145/3292500.3330840) (KDD 2019) [:octocat:](https://github.com/dengbaowang/PL-AGGD)
- [Partial Label Learning by Semantic Difference Maximization](https://www.ijcai.org/proceedings/2019/318) (IJCAI 2019)
- [Partial Label Learning with Self-Guided Retraining](https://arxiv.org/abs/1902.03045) (AAAI 2019) [:octocat:](https://lfeng1995.github.io/Code/SURE.zip)
- [Partial Label Learning via Label Enhancement](https://ojs.aaai.org/index.php/AAAI/article/view/4497) (AAAI 2019)
- [GM-PLL: Graph Matching based Partial Label Learning](https://arxiv.org/abs/1901.03073) (TKDE 2019)

<a name="instance-dependent-pll" />

### <u>Instance-Dependent Partial-Label Learning</u>

- [Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Mitigating_Instance_Entanglement_in_Instance-Dependent_Partial_Label_Learning_CVPR_2026_paper.html) (CVPR 2026) [:octocat:](https://github.com/RyanZhaoIc/CAD)
- [From Predictions to Embeddings: Dual Knowledge Distillation for Instance-Dependent Partial Label Learning](https://doi.org/10.1007/978-3-032-37624-4_23) (ECCV 2026)
- [Mixed Blessing: Class-Wise Embedding Guided Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2412.05029) (KDD 2025) [:octocat:](https://github.com/Yangfc-ML/CEL)
- [Reduction-based Pseudo-label Generation for Instance-dependent Partial Label Learning](https://arxiv.org/abs/2410.20797) (NeurIPS 2025)
- [Distilling Reliable Knowledge for Instance-Dependent Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29519) (AAAI 2024) [:octocat:](https://github.com/wu-dd/DIRK)
- [Variational Label Enhancement for Instance-Dependent Partial Label Learning](https://doi.org/10.1109/TPAMI.2024.3455260) (TPAMI 2024) [:octocat:](https://github.com/palm-ml/valen)
- [Candidate-aware Selective Disambiguation Based On Normalized Entropy for Instance-dependent Partial-label Learning](https://ieeexplore.ieee.org/document/10376678) (ICCV 2023)
- [Decompositional Generation Process for Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2204.03845) (ICLR 2023) [:octocat:](https://github.com/palm-ml/idgp)
- [Progressive Purification for Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2206.00830) (ICML 2023) [:octocat:](https://github.com/palm-ml/POP)
- [Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning](https://www.ijcai.org/proceedings/2022/502) (IJCAI 2022) [:octocat:](https://github.com/eddiexia98/ABLE)
- [Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2110.12911) (NeurIPS 2021) [:octocat:](https://github.com/palm-ml/valen)

<a name="foundation-models" />

### <u>Learning with Foundation / Vision-Language Models</u>

- [Weakly Supervised Classification with Pre-Trained Models: A Robust Fine-Tuning Approach](https://doi.org/10.1007/s10994-025-06966-z) (ML 2026)
- [Holistic Optimal Label Selection for Robust Prompt Learning under Partial Labels](https://doi.org/10.1007/978-3-032-37592-6_27) (ECCV 2026) [:octocat:](https://github.com/Qizhoay/HopS)
- [Tuning the Right Foundation Models is What You Need for Partial Label Learning](https://arxiv.org/abs/2506.05027) (arXiv 2025) [:octocat:](https://github.com/SEU-hk/PartialCLIP)
- [Pre-Trained Vision-Language Models as Noisy Partial Annotators](https://ojs.aaai.org/index.php/AAAI/article/view/35417) (AAAI 2025) [:octocat:](https://github.com/Portugas0807/Co-Reg)
- [Learning from True-False Labels via Multi-modal Prompt Retrieving](https://proceedings.mlr.press/v267/li25dl.html) (ICML 2025) [:octocat:](https://github.com/Tranquilxu/TMP)
- [Candidate Pseudolabel Learning: Enhancing Vision-Language Models by Prompt Tuning with Unlabeled Data](https://proceedings.mlr.press/v235/zhang24bo.html) (ICML 2024) [:octocat:](https://github.com/vanillaer/CPL-ICML2024)

<a name="complementary-label-learning" />

### <u>Complementary-Label Learning Methods</u>

- [Learning from Stochastic Labels](https://arxiv.org/abs/2302.00299) (ICME 2025)
- [Complementary Label Learning with Positive Label Guessing and Negative Label Enhancement](https://openreview.net/forum?id=LPRxGZ7Oax) (ICLR 2025) [:octocat:](https://github.com/yhli-ml/PLNL)
- [Self-Adaptive Label Discovery and Multi-View Fusion for Complementary Label Learning](https://doi.org/10.1016/j.neunet.2024.106763) (NN 2025)
- [ComCo: Complementary Supervised Contrastive Learning for Complementary Label Learning](https://doi.org/10.1016/j.neunet.2023.10.013) (NN 2024)
- [Enhancing Label Sharing Efficiency in Complementary-Label Learning with Label Augmentation](https://arxiv.org/abs/2305.08344) (arXiv 2023)
- [Biased Complementary-Label Learning Without True Labels](https://ieeexplore.ieee.org/document/9836971) (TNNLS 2022)
- [Discriminative Complementary-Label Learning with Weighted Loss](https://proceedings.mlr.press/v139/gao21d.html) (ICML 2021) [:octocat:](https://github.com/gaoyi439/complementary-label-learning)
- [Learning from Complementary Labels via Partial-Output Consistency Regularization](https://www.ijcai.org/proceedings/2021/423) (IJCAI 2021) [:octocat:](https://palm.seu.edu.cn/zhangml/files/POCR.zip)
- [Generative-Discriminative Complementary Learning](https://arxiv.org/abs/1904.01612) (AAAI 2020) [:octocat:](https://github.com/xuyanwu/Complementary-GAN)

<a name="multi-complementary-label-learning" />

### <u>Multiple / Multi-Label Complementary-Label Learning</u>

- [Data Augmentation with Regularization for Multi-Labeled Complementary Label Learning](https://ieeexplore.ieee.org/document/11222868) (TPAMI 2026) [:octocat:](https://github.com/gaoyi439/NMCB)
- [ComRank: Ranking Loss for Multi-Label Complementary Label Learning](https://neurips.cc/virtual/2025/poster/115492) (NeurIPS 2025) [:octocat:](https://github.com/JellyJamZhu/ComRank)
- [Multi-Label Learning with Multiple Complementary Labels](https://ieeexplore.ieee.org/document/11016060) (TPAMI 2025) [:octocat:](https://github.com/gaoyi439/CTL)
- [Complementary to Multiple Labels: A Correlation-Aware Correction Approach](https://ieeexplore.ieee.org/document/10561516) (TPAMI 2024) [:octocat:](https://github.com/GaoYi439/MLCL)
- [Unbiased Risk Estimator to Multi-Labeled Complementary Label Learning](https://www.ijcai.org/proceedings/2023/415) (IJCAI 2023) [:octocat:](https://github.com/gaoyi439/GDF)
- [Multi-Complementary and Unlabeled Learning for Arbitrary Losses and Models](https://arxiv.org/abs/2001.04243) (PR 2022)
- [Learning with Multiple Complementary Labels](https://arxiv.org/abs/1912.12927) (ICML 2020)

<a name="dimensionality-reduction" />

### <u>Dimensionality Reduction & Feature Selection</u>

- [Granular Ball-Guided Disambiguation for Partial Multilabel Feature Selection via Maximum Consistency Minimum Uncertainty](https://doi.org/10.1109/TNNLS.2025.3610795) (TNNLS 2026)
- [Partial multi-label feature selection via dual-space combination and disambiguation strategy fusion](https://doi.org/10.1016/j.patcog.2026.113537) (PR 2026)
- [Reliability-enhanced partial multi-label feature selection based on neighborhood rough set](https://doi.org/10.1016/j.patcog.2026.113762) (PR 2026)
- [CLASS: Deep Partial Label Feature Selection with Cluster-Guided Disambiguation and Structured Sparsity](https://doi.org/10.1145/3770854.3780163) (KDD 2026)
- [Partial Label Feature Selection with Dynamic Streaming Labels](https://doi.org/10.1016/j.patcog.2025.112650) (PR 2026)
- [Wrapped Partial Label Dimensionality Reduction via Dependence Maximization](https://www.ijcai.org/proceedings/2025/769) (IJCAI 2025) [:octocat:](https://palm.seu.edu.cn/zhangml/files/WPLDR.rar)
- [Partial multi-label feature selection with feature noise](https://doi.org/10.1016/j.patcog.2024.111310) (PR 2025) [:octocat:](https://github.com/View5U/Partial-Multi-Label-Feature-Selection-with-Feature-Noise)
- [Integrating label confidence-based feature selection for partial multi-label learning](https://doi.org/10.1016/j.patcog.2024.111281) (PR 2025)
- [Partial multi-label feature selection based on label distribution learning](https://doi.org/10.1016/j.patcog.2025.111523) (PR 2025)
- [Dimensionality Reduction for Partial Label Learning: A Unified and Adaptive Approach](https://ieeexplore.ieee.org/document/10440495) (TKDE 2024) [:octocat:](https://palm.seu.edu.cn/zhangml/files/DRAW.zip)
- [Partial Label Feature Selection: An Adaptive Approach](https://doi.org/10.1109/tkde.2024.3365691) (TKDE 2024)
- [Confidence-Induced Granular Partial Label Feature Selection via Dependency and Similarity](https://doi.org/10.1109/tkde.2024.3405489) (TKDE 2024)
- [Partial Label Feature Selection Based on Noisy Manifold and Label Distribution](https://doi.org/10.1016/j.patcog.2024.110791) (PR 2024)
- [Disambiguation Enabled Linear Discriminant Analysis for Partial Label Dimensionality Reduction](https://dl.acm.org/doi/abs/10.1145/3494565) (TKDD 2022) [:octocat:](https://palm.seu.edu.cn/zhangml/files/DELIN.rar)
- [Submodular Feature Selection for Partial Label Learning](https://dl.acm.org/doi/abs/10.1145/3534678.3539292) (KDD 2022) [:octocat:](https://palm.seu.edu.cn/zhangml/files/SAUTE.rar)
- [Partial Label Dimensionality Reduction via Confidence-Based Dependence Maximization](https://dl.acm.org/doi/abs/10.1145/3447548.3467313) (KDD 2021) [:octocat:](https://palm.seu.edu.cn/zhangml/files/CENDA.zip)

<a name="data-augmentation" />

### <u>Data Augmentation</u>

- [Making Partial-Label Datasets Easier: A Simple Yet Highly Effective Data Augmentation for Deep Partial-Label Learning](https://doi.org/10.1007/978-3-032-37372-4_28) (ECCV 2026)
- [Intra-Cluster Mixup: An Effective Data Augmentation Technique for Complementary-Label Learning](https://openreview.net/forum?id=h9PbmfznWj) (TMLR 2026)
- [Partial Label Learning with Discrimination Augmentation](https://dl.acm.org/doi/abs/10.1145/3534678.3539363) (KDD 2022) [:octocat:](https://github.com/wwangwitsel/PLDA)

<a name="problem-settings-extensions" />

## <u>Problem Settings & Extensions</u>

<a name="partial-multi-label-learning" />

### <u>Partial Multi-Label Learning</u>

- [Rethinking Consistent Multi-Label Classification Under Inexact Supervision](https://arxiv.org/abs/2510.04091) (ICLR 2026)
- [Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge](https://arxiv.org/abs/2507.05992) (TMM 2026) [:octocat:](https://github.com/wuliwuxin/SCINet)
- [Revisiting Sparsity Constraint Under High-Rank Property in Partial Multi-Label Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Si_Revisiting_Sparsity_Constraint_Under_High-Rank_Property_in_Partial_Multi-Label_Learning_CVPR_2026_paper.html) (CVPR 2026) [:octocat:](https://github.com/cuiyidan0514/Schirn)
- [Label Confidence Recovery with High-order Label Correlation in Partial Multi-label Learning](https://www.ijcai.org/proceedings/2026/472) (IJCAI 2026)
- [Exploiting Global Information for Partial Multi-Label Learning](https://doi.org/10.1109/TKDE.2026.3677879) (TKDE 2026)
- [Noise-Resistant Candidate Label Generation Decomposition for Instance-Dependent Partial Multilabel Learning](https://doi.org/10.1109/TNNLS.2026.3742440) (TNNLS 2026) [:octocat:](https://github.com/zhongjingyu1/NRGD)
- [Learning embedded label-specific features for partial multi-label learning](https://doi.org/10.1016/j.patcog.2025.112005) (PR 2026)
- [Contrastive flow with controllable augmentation for partial multi-label learning](https://doi.org/10.1016/j.patcog.2026.113959) (PR 2026)
- [Dual feature-driven approach for partial multi-label learning](https://doi.org/10.1016/j.patcog.2026.114433) (PR 2026)
- [Embracing sparse features: Partial Multi-Label Learning via noisy label identification](https://doi.org/10.1016/j.patcog.2026.113756) (PR 2026)
- [FastGRAIL: Anchor-graph-based fast adaptive partial multi-label learning with label correlations](https://doi.org/10.1016/j.patcog.2026.113457) (PR 2026)
- [Partial multi-label learning with discriminative feature subspace and dynamic label correlation](https://doi.org/10.1016/j.patcog.2026.113543) (PR 2026)
- [Noise correction and distribution fine-tuning for long-tailed partial multi-label learning](https://doi.org/10.1016/j.patcog.2026.114096) (PR 2026) [:octocat:](https://github.com/zhongjingyu1/NDLP)
- [Noise-Resistant Label Reconstruction Feature Selection for Partial Multi-Label Learning](https://www.ijcai.org/proceedings/2025/576) (IJCAI 2025) [:octocat:](https://github.com/typsdfgh/PML-FSMIR)
- [Partial Multilabel Learning Using Noise-Tolerant Broad Learning System With Label Enhancement and Dimensionality Reduction](https://ieeexplore.ieee.org/document/10416802) (TNNLS 2025)
- [Confidence-Aware With Prototype Alignment for Partial Multi-label Learning](https://openreview.net/forum?id=Hp78SvTU2N) (NeurIPS 2025)
- [Calibrated Disambiguation for Partial Multi-label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/34049) (AAAI 2025)
- [Reconsidering Feature Structure Information and Latent Space Alignment in Partial Multi-label Feature Selection](https://ojs.aaai.org/index.php/AAAI/article/view/34179) (AAAI 2025)
- [Pseudo-Label Reconstruction for Partial Multi-Label Learning](https://www.ijcai.org/proceedings/2025/545) (IJCAI 2025) [:octocat:](https://github.com/CcAmbiguous/PML-PLR)
- [Partial multi-label learning via label-specific feature corrections](https://doi.org/10.1007/s11432-023-4230-2) (SCIS 2025)
- [Fuzzy bifocal disambiguation for partial multi-label learning](https://doi.org/10.1016/j.neunet.2025.107137) (NN 2025)
- [VADIS: Investigating Inter-View Representation Biases for Multi-View Partial Multi-Label Learning](https://proceedings.mlr.press/v286/wang25e.html) (UAI 2025) [:octocat:](https://github.com/palm-ml/vadis)
- [A Deep Model for Partial Multi-Label Image Classification with Curriculum Based Disambiguation](https://arxiv.org/abs/2207.02410) (MIR 2024) [:octocat:](https://github.com/lee-plus-plus/PML-CDCR)
- [WPML3CP: Wasserstein Partial Multi-Label Learning with Dual Label Correlation Perspectives](https://www.ijcai.org/proceedings/2024/495) (IJCAI 2024)
- [Noisy Label Removal for Partial Multi-Label Learning](https://doi.org/10.1145/3637528.3671677) (KDD 2024) [:octocat:](https://github.com/Yangfc-ML/NLR)
- [Partial Multi-label Learning Based On Near-Far Neighborhood Label Enhancement And Nonlinear Guidance](https://openreview.net/forum?id=qZwp41zYjF) (ACM MM 2024) [:octocat:](https://github.com/CcAmbiguous/PML-LENFN)
- [Learning Accurate Label-Specific Features From Partially Multilabeled Data](https://doi.org/10.1109/TNNLS.2023.3241921) (TNNLS 2024)
- [Limited-Supervised Multi-Label Learning with Dependency Noise](https://ojs.aaai.org/index.php/AAAI/article/view/29494) (AAAI 2024)
- [Towards Enabling Binary Decomposition for Partial Multi-Label Learning](https://ieeexplore.ieee.org/document/10168295) (TPAMI 2023)
- [Deep Partial Multi-Label Learning with Graph Disambiguation](https://arxiv.org/abs/2305.05882) (IJCAI 2023)
- [Partial Multi-Label Learning with Probabilistic Graphical Disambiguation](https://papers.nips.cc/paper_files/paper/2023/hash/04e05ba5cbc36044f6499d1edf15247e-Abstract-Conference.html) (NeurIPS 2023) [:octocat:](https://palm.seu.edu.cn/zhangml/files/PARD.rar)
- [ProPML: Probability Partial Multi-label Learning](https://arxiv.org/abs/2403.07603) (DSAA 2023) [:octocat:](https://github.com/gmum/propml)
- [Prior Knowledge Regularized Self-Representation Model for Partial Multilabel Learning](https://ieeexplore.ieee.org/document/9533180) (TCYB 2021)
- [Global-Local Label Correlation for Partial Multi-Label Learning](https://ieeexplore.ieee.org/document/9343691) (TMM 2021)
- [Progressive Enhancement of Label Distributions for Partial Multilabel Learning](https://ieeexplore.ieee.org/document/9615493) (TNNLS 2021)
- [Partial Multi-Label Learning With Noisy Label Identification](https://ieeexplore.ieee.org/document/9354590) (TPAMI 2021)
- [Partial Multi-Label Learning via Credible Label Elicitation](https://ieeexplore.ieee.org/document/9057438) (TPAMI 2021) [:octocat:](https://palm.seu.edu.cn/zhangml/files/PARTICLE.rar)
- [Partial Multi-Label Learning with Meta Disambiguation](https://dl.acm.org/doi/abs/10.1145/3447548.3467259) (KDD 2021)
- [Understanding Partial Multi-Label Learning via Mutual Information](https://proceedings.neurips.cc/paper/2021/hash/217c0e01c1828e7279051f1b6675745d-Abstract.html) (NeurIPS 2021)
- [Multi-View Partial Multi-Label Learning with Graph-Based Disambiguation](https://ojs.aaai.org/index.php/AAAI/article/view/5761) (AAAI 2020) [:octocat:](https://palm.seu.edu.cn/zhangml/files/GRADIS.rar)
- [Partial Multi-Label Learning via Multi-Subspace Representation](https://www.ijcai.org/proceedings/2020/362) (IJCAI 2020)
- [Feature-Induced Manifold Disambiguation for Multi-View Partial Multi-label Learning](https://dl.acm.org/doi/10.1145/3394486.3403098) (KDD 2020) [:octocat:](https://palm.seu.edu.cn/zhangml/files/FIMAN.rar)

<a name="multi-label-partial-annotations" />

### <u>Multi-Label Learning with Partial Annotations</u>

- [LogicMix: Sample Mixing Data Augmentation for Multi-Label Image Classification with Partial Labels](https://doi.org/10.1016/j.patcog.2025.112186) (PR 2026)
- [View Confidence Perception-Driven Incremental Prediction for Incomplete Multi-view Multi-label Learning](https://openreview.net/forum?id=WBWPvuJd9B) (NeurIPS 2026) [:octocat:](https://github.com/justsmart/VCPIP)
- [Label-Guided Representation Learning for Incomplete Multi-View Multi-Label Classification](https://openreview.net/forum?id=cY2nPckqzQ) (ICML 2026)
- [Permutation-Consistent Variational Encoding for Incomplete Multi-View Multi-Label Classification](https://openreview.net/forum?id=y4LyiOIOUn) (ICLR 2026) [:octocat:](https://github.com/justsmart/PCVE)
- [Incomplete Multi-View Multi-Label Classification via Shared Codebook and Fused-Teacher Self-Distillation](https://openreview.net/forum?id=LAuep7N7rF) (ICLR 2026) [:octocat:](https://github.com/xuy11/SCSD)
- [Cross-View Distillation and Adaptive Masking for Incomplete Multi-View Multi-Label Classification](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Cross-View_Distillation_and_Adaptive_Masking_for_Incomplete_Multi-View_Multi-Label_Classification_CVPR_2026_paper.html) (CVPR 2026)
- [Partial Multiview Incomplete Multilabel Learning via Uncertainty-Driven Reliable Dynamic Fusion](https://doi.org/10.1109/TPAMI.2025.3603677) (TPAMI 2026) [:octocat:](https://github.com/longjianghhh/URDF)
- [Multi-Domain Feature Integration Based Trusted Partial Multi-View Incomplete Multi-Label Learning](https://doi.org/10.1109/TPAMI.2026.3692653) (TPAMI 2026) [:octocat:](https://github.com/Jocelyn01-z/MFIT)
- [Learning Compact Semantic Information and Reliable Pseudo-Labels for Incomplete Multi-View Multi-Label Classification](https://doi.org/10.1109/TPAMI.2026.3665813) (TPAMI 2026)
- [Disentangling Consistent and Specific Information for Double Incomplete Multi-View Multi-Label Classification](https://doi.org/10.1109/TPAMI.2026.3665097) (TPAMI 2026) [:octocat:](https://github.com/daguankuaile-lang/DCSI)
- [When Semantically Consistent Encoding Meets View-Label Heterogeneity Modeling: A Unified Framework for Incomplete Multi-View Multi-Label Learning](https://doi.org/10.1109/TPAMI.2026.3728832) (TPAMI 2026) [:octocat:](https://github.com/justsmart/V2L)
- [Semantic Invariant Representation Learning and Data-Driven Label Prototype Modeling for Incomplete Multi-view Multi-label Classification](https://doi.org/10.1007/s11263-026-03025-5) (IJCV 2026) [:octocat:](https://github.com/justsmart/SIPOT)
- [Adapting Dense Vision-Language Relationships for Multi-Label Classification With Partial Label](https://doi.org/10.1109/TPAMI.2026.3706734) (TPAMI 2026) [:octocat:](https://github.com/iCVTEAM/LDSA)
- [One Coin Has Two Sides: Single Positive Multi Label Learning from Salient Annotations](https://openreview.net/forum?id=LhhFsfAzL6) (ICML 2026) [:octocat:](https://github.com/RainyWangxy/PiSA)
- [Exploiting Negative Multi-Cluster Structure in Class-Wise Embeddings for Weakly Supervised Multi-Label Learning](https://openreview.net/forum?id=Jc0lO9zGdF) (NeurIPS 2026)
- [Single positive multi-label learning via label ranking relationship](https://doi.org/10.1016/j.patcog.2026.113542) (PR 2026)
- [PositiveCoOp: Rethinking Prompting Strategies for Multi-Label Recognition with Partial Annotations](https://arxiv.org/abs/2409.08381) (WACV 2025) [:octocat:](https://github.com/SamyakR99/PositiveCoOp)
- [Reliable Representation Learning for Incomplete Multi-View Missing Multi-Label Classification](https://arxiv.org/abs/2303.17117) (TPAMI 2025) [:octocat:](https://github.com/justsmart/RANK)
- [Theory-Driven Label-Specific Representation for Incomplete Multi-View Multi-Label Learning](https://openreview.net/forum?id=0Az25lvdT2) (NeurIPS 2025)
- [Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label Classification](https://proceedings.mlr.press/v267/wen25c.html) (ICML 2025)
- [Incomplete Multi-View Multi-Label Learning via Disentangled Representation and Label Semantic Embedding](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Incomplete_Multi-View_Multi-label_Learning_via_Disentangled_Representation_and_Label_Semantic_CVPR_2025_paper.html) (CVPR 2025) [:octocat:](https://github.com/xuy11/DRLS)
- [Dual Structure-guided Contrastive Network for Incomplete Multi-view Partial Multi-label Classification](https://doi.org/10.1145/3711896.3736916) (KDD 2025)
- [Addressing Multi-Label Learning with Partial Labels: From Sample Selection to Label Selection](https://ojs.aaai.org/index.php/AAAI/article/view/34119) (AAAI 2025)
- [Prompt-guided consistency learning for multi-label classification with incomplete labels](https://doi.org/10.1016/j.neunet.2025.107604) (NN 2025)
- [Instance-Dependent Incomplete Multi-Label Feature Selection by Fuzzy Tolerance Relation and Fuzzy Mutual Implication Granularity](https://doi.org/10.1109/TKDE.2025.3591461) (TKDE 2025)
- [Can Class-Priors Help Single-Positive Multi-Label Learning?](https://openreview.net/forum?id=9z6ZzeKqPC) (NeurIPS 2025)
- [More Reliable Pseudo-labels, Better Performance: A Generalized Approach to Single Positive Multi-label Learning](https://openaccess.thecvf.com/content/ICCV2025/html/Tran_More_Reliable_Pseudo-labels_Better_Performance_A_Generalized_Approach_to_Single_ICCV_2025_paper.html) (ICCV 2025) [:octocat:](https://github.com/Fsoft-AIC/AEVLP)
- [Single-positive Multi-label Learning with Label Cardinality](https://openreview.net/forum?id=XEPPXH2nKu) (TMLR 2025) [:octocat:](https://github.com/shayangharib/SPMLL_with_Label_Cardinality)
- [Pushing one pair of labels apart each time in multi-label learning: from single positive to full labels](https://doi.org/10.1007/s11432-023-3979-9) (SCIS 2025)
- [Partial Multi-View Multi-Label Classification via Semantic Invariance Learning and Prototype Modeling](https://openreview.net/forum?id=5ap1MmUqO6) (ICML 2024) [:octocat:](https://github.com/justsmart/SIP)
- [Combining Supervised Learning and Reinforcement Learning for Multi-Label Classification Tasks with Partial Labels](https://aclanthology.org/2024.acl-long.731/) (ACL 2024)
- [View-Category Interactive Sharing Transformer for Incomplete Multi-View Multi-Label Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Ou_View-Category_Interactive_Sharing_Transformer_for_Incomplete_Multi-View_Multi-Label_Learning_CVPR_2024_paper.html) (CVPR 2024)
- [DualCoOp++: Fast and Effective Adaptation to Multi-Label Recognition With Limited Annotations](https://doi.org/10.1109/TPAMI.2023.3346405) (TPAMI 2024)
- [Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels](https://doi.org/10.1007/s11263-024-02127-2) (IJCV 2024) [:octocat:](https://github.com/HCPLab-SYSU/HCP-MLR-PL)
- [Text-Region Matching for Multi-Label Image Recognition with Missing Labels](https://openreview.net/forum?id=XHaqiDP293) (ACM MM 2024) [:octocat:](https://github.com/yu-gi-oh-leilei/TRM-ML)
- [Multi-Label Learning with Block Diagonal Labels](https://openreview.net/forum?id=VljEshSNa6) (ACM MM 2024)
- [Adaptive ambiguity-aware weighting for multi-label recognition with limited annotations](https://doi.org/10.1016/j.neunet.2024.106642) (NN 2024)
- [Feature selection for multilabel classification with missing labels via multi-scale fusion fuzzy uncertainty measures](https://doi.org/10.1016/j.patcog.2024.110580) (PR 2024)
- [IGNORE: Information Gap-based False Negative Loss Rejection for Single Positive Multi-Label Learning](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5081_ECCV_2024_paper.php) (ECCV 2024) [:octocat:](https://github.com/gyeong-ryeol-song/SPML-IGNORE)
- [Boosting Single Positive Multi-label Classification with Generalized Robust Loss](https://www.ijcai.org/proceedings/2024/423) (IJCAI 2024) [:octocat:](https://github.com/yan4xi1/GRLoss)
- [Deep Double Incomplete Multi-View Multi-Label Learning With Incomplete Labels and Missing Views](https://ieeexplore.ieee.org/document/10086538) (TNNLS 2023) [:octocat:](https://github.com/justsmart/DIMC)
- [Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels](https://arxiv.org/abs/2203.02172) (AAAI 2022) [:octocat:](https://github.com/HCPLab-SYSU/HCP-MLR-PL)
- [Structured Semantic Transfer for Multi-Label Recognition with Partial Labels](https://arxiv.org/abs/2112.10941) (AAAI 2022) [:octocat:](https://github.com/HCPLab-SYSU/HCP-MLR-PL)
- [Multi-label Classification with Partial Annotations using Class-aware Selective Loss](https://arxiv.org/abs/2110.10955) (CVPR 2022) [:octocat:](https://github.com/Alibaba-MIIL/PartialLabelingCSL)
- [PLMCL: Partial-Label Momentum Curriculum Learning for Multi-Label Image Classification](https://arxiv.org/abs/2208.09999) (ECCVw 2022)
- [Learning a Deep ConvNet for Multi-label Classification with Partial Labels](https://arxiv.org/abs/1902.09720) (CVPR 2019)

<a name="noisy-partial-label-learning" />

### <u>Noisy Partial-Label Learning</u>

- [Neighbor-aware Label Refinement: Enhancing Unreliable Instance-Dependent Partial Labels](https://ojs.aaai.org/index.php/AAAI/article/view/39784) (AAAI 2026) [:octocat:](https://github.com/TangXJ-895/NLAP)
- [Bridging Weakly-Supervised Learning and VLM Distillation: Noisy Partial Label Learning for Efficient Downstream Adaptation](https://arxiv.org/abs/2506.03229) (arXiv 2025) [:octocat:](https://github.com/Portugas0807/Co-Reg)
- [Noise Separation guided Candidate Label Reconstruction for Noisy Partial Label Learning](https://openreview.net/forum?id=TOahfjA3sP) (ICLR 2025) [:octocat:](https://github.com/pruirui/PLRC)
- [Pseudo-labelling meets Label Smoothing for Noisy Partial Label Learning](https://arxiv.org/abs/2402.04835) (CVPRW 2025) [:octocat:](https://github.com/darshana1406/PALS)
- [IRNet: Iterative Refinement Network for Noisy Partial Label Learning](https://arxiv.org/abs/2211.04774) (TPAMI 2025) [:octocat:](https://github.com/zeroQiaoba/IRNet)
- [Sample Selection for Noisy Partial Label Learning with Interactive Contrastive Learning](https://doi.org/10.1016/j.patcog.2025.111681) (PR 2025)
- [PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning](https://arxiv.org/abs/2201.08984) (TPAMI 2023) [:octocat:](https://github.com/hbzju/PiCO)
- [On the Robustness of Average Losses for Partial-Label Learning](https://arxiv.org/abs/2106.06152) (TPAMI 2023)
- [FREDIS: A Fusion Framework of Refinement and Disambiguation for Unreliable Partial Label Learning](https://proceedings.mlr.press/v202/qiao23b.html) (ICML 2023) [:octocat:](https://github.com/palm-ml/fredis)
- [Unreliable Partial Label Learning with Recursive Separation](https://arxiv.org/abs/2302.09891) (IJCAI 2023)
- [ALIM: Adjusting Label Importance Mechanism for Noisy Partial Label Learning](https://arxiv.org/abs/2301.12077) (NeurIPS 2023) [:octocat:](https://github.com/zeroQiaoba/ALIM)
- [Mutual Partial Label Learning with Competitive Label Noise](https://openreview.net/forum?id=EUrxG8IBCrC) (ICLR 2023)
- [Robust Representation Learning for Unreliable Partial Label Learning](https://arxiv.org/abs/2308.16718) (arXiv 2023) [:octocat:](https://github.com/wu-dd/URRL)
- [Learning with Noisy Partial Labels by Simultaneously Leveraging Global and Local Consistencies](https://dl.acm.org/doi/10.1145/3340531.3411885) (CIKM 2020) [:octocat:](https://github.com/changchunli/PANGOLIN)

<a name="semi-supervised-partial-label-learning" />

### <u>Semi-Supervised Partial-Label Learning</u>

- [Semi-Supervised Partial Label Learning via Label Confidence Recovery](https://doi.org/10.1016/j.patcog.2026.113387) (PR 2026)
- [Partial Label Learning with Semi-Supervised Clustering Disambiguation](https://doi.org/10.1016/j.patcog.2025.112138) (PR 2026) [:octocat:](https://github.com/liujunying-ai/PLOD)
- [Collaborative Dual Representations for Semi-Supervised Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/39049) (AAAI 2026)
- [AdapMatch: Adaptive Bias Decoupling for Semi-Supervised Partial Label Learning under Unknown Class Distributions](https://openreview.net/forum?id=eAtYtntHfU) (NeurIPS 2026)
- [Learning with Partial-Label and Unlabeled Data: A Uniform Treatment for Supervision Redundancy and Insufficiency](https://proceedings.mlr.press/v235/liu24ar.html) (ICML 2024) [:octocat:](https://github.com/livyf/SPMI)
- [FairMatch: Promoting Partial Label Learning by Unlabeled Samples](https://dl.acm.org/doi/abs/10.1145/3637528.3671685) (KDD 2024) [:octocat:](https://github.com/jhjiangSEU/FairMatch)
- [Controller-Guided Partial Label Consistency Regularization with Unlabeled Data](https://ojs.aaai.org/index.php/AAAI/article/view/29484) (AAAI 2024)
- [Learning with Partial Labels from Semi-supervised Perspective](https://arxiv.org/abs/2211.13655) (AAAI 2023) [:octocat:](https://github.com/changchunli/PLSP)
- [Distributed Semisupervised Partial Label Learning Over Networks](https://ieeexplore.ieee.org/document/9699063) (TAI 2022)
- [Exploiting Unlabeled Data via Partial Label Assignment for Multi-Class Semi-Supervised Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17310) (AAAI 2021) [:octocat:](https://palm.seu.edu.cn/zhangml/files/EUPAL.zip)
- [Semi-Supervised Partial Label Learning via Confidence-Rated Margin Maximization](https://proceedings.neurips.cc/paper/2020/hash/4dea382d82666332fb564f2e711cbc71-Abstract.html) (NeurIPS 2020) [:octocat:](https://palm.seu.edu.cn/zhangml/files/PARM.zip)
- [Partial Label Learning with Unlabeled Data](https://www.ijcai.org/proceedings/2019/521) (IJCAI 2019)

<a name="multi-instance-learning" />

### <u>Multi-Instance Partial-Label Learning</u>

- [Calibratable Disambiguation Loss for Multi-Instance Partial-Label Learning](https://arxiv.org/abs/2512.17788) (TPAMI 2026) [:octocat:](https://github.com/tangw-seu/MIPLCDL)
- [Dual Graph Disambiguation for Multi-Instance Partial-Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/40161) (AAAI 2026) [:octocat:](https://github.com/hzzhuzhen/DualG)
- [Enhanced Multi-Instance Partial Label Learning via Average Gradient Outer Product](https://proceedings.mlr.press/v306/cao26af.html) (ICML 2026)
- [Hybrid Feature-Based Collaborative Disambiguation for Multi-Instance Partial-Label Learning](https://doi.org/10.1016/j.neunet.2026.109648) (NN 2026)
- [Fast Multi-Instance Partial-Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/34356) (AAAI 2025) [:octocat:](https://github.com/yangyf22/FastMIPL)
- [Imbalances in Neurosymbolic Learning: Characterization and Mitigating Strategies](https://papers.nips.cc/paper_files/paper/2025/hash/df46fcdb724ab86d36e5822b4d657ea9-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/tsamoura/imbalances-nsl)
- [Multi-Instance Partial-Label Learning with Margin Adjustment](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2e5a74735e5ada01d85afbb49caad36f-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/tangw-seu/MIPLMA)
- [Exploiting Conjugate Label Information for Multi-Instance Partial-Label Learning](https://arxiv.org/abs/2408.14369) (IJCAI 2024) [:octocat:](https://github.com/tangw-seu/ELIMIPL)
- [Multi-Instance Partial-Label Learning: Towards Exploiting Dual Inexact Supervision](https://arxiv.org/abs/2212.08997) (SCIS 2024) [:octocat:](https://github.com/tangw-seu/MIPLGP)
- [Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning](https://arxiv.org/abs/2305.16912) (NeurIPS 2023) [:octocat:](https://github.com/tangw-seu/DEMIPL)

<a name="class-imbalance-problem" />

### <u>Imbalanced Partial-Label Learning</u>

- [Class-Prior Perturbation-Robust Regularization for Imbalanced Unreliable Partial Label Learning](https://openreview.net/forum?id=8xyaIv2OKq) (ICML 2026)
- [Dual Branch Mutual Teaching for Long-Tailed Partial Label Learning](https://www.ijcai.org/proceedings/2026/528) (IJCAI 2026) [:octocat:](https://github.com/YukiNozzzz/DEACON)
- [GBRIP: Granular Ball Representation for Imbalanced Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/33916) (AAAI 2025)
- [Long-Tailed Partial Label Learning by Head Classifier and Tail Classifier Cooperation](https://ojs.aaai.org/index.php/AAAI/article/view/29182) (AAAI 2024) [:octocat:](https://github.com/pruirui/HTC-LTPLL)
- [Pseudo Labels Regularization for Imbalanced Partial-Label Learning](https://arxiv.org/abs/2303.03946) (ICASSP 2024)
- [Learning in Imperfect Environment: Multi-Label Classification with Long-Tailed Distribution and Partial Labels](https://arxiv.org/abs/2304.10539) (ICCV 2023) [:octocat:](https://github.com/wannature/COMIC)
- [Class-Imbalanced Complementary-Label Learning via Weighted Loss](https://arxiv.org/abs/2209.14189) (NN 2023)
- [Long-Tailed Partial Label Learning via Dynamic Rebalancing](https://arxiv.org/abs/2302.05080) (ICLR 2023) [:octocat:](https://github.com/MediaBrain-SJTU/RECORDS-LTPLL)
- [SoLar: Sinkhorn Label Refinery for Imbalanced Partial-Label Learning](https://arxiv.org/abs/2209.10365) (NeurIPS 2022) [:octocat:](https://github.com/hbzju/SoLar)
- [A Partial Label Metric Learning Algorithm for Class Imbalanced Data](https://proceedings.mlr.press/v157/liu21f.html) (ACML 2021)
- [Towards Mitigating the Class-Imbalance Problem for Partial Label Learning](https://dl.acm.org/doi/10.1145/3219819.3220008) (KDD 2018) [:octocat:](https://github.com/seu71wj/CIMAP)

<a name="open-set-problem" />

### <u>Open-Set & Out-of-Distribution Partial-Label Learning</u>

- [PLOOD: Partial Label Learning with Out-of-distribution Objects](https://arxiv.org/abs/2403.06681) (arXiv 2024)
- [Out-of-Distribution Detection by Cross-Class Vicinity Distribution of In-Distribution Data](https://arxiv.org/abs/2206.09385) (TNNLS 2024) [:octocat:](https://github.com/Lawliet-zzl/LCVD)
- [Partial-label Learning with Mixed Closed-set and Open-set Out-of-candidate Examples](https://arxiv.org/abs/2307.00553) (KDD 2023) [:octocat:](https://github.com/shuohe1995/OOC_PLL)

<a name="incremental-learning" />

### <u>Incremental Learning & Augmented Classes</u>

- [Towards Robust Incremental Learning Under Ambiguous Supervision](https://www.ijcai.org/proceedings/2025/712) (IJCAI 2025)
- [Model Rectification With Simultaneous Incremental Feature and Partial Label Set](https://doi.org/10.1109/TPAMI.2025.3600033) (TPAMI 2025)
- [An Unbiased Risk Estimator for Partial Label Learning with Augmented Classes](https://arxiv.org/abs/2409.19600) (TIST 2024) [:octocat:](https://github.com/hujiayu1223/PLLAC-project)
- [Complementary Labels Learning with Augmented Classes](https://arxiv.org/abs/2211.10701) (KBS 2024)
- [Partial label learning with emerging new labels](https://link.springer.com/article/10.1007/s10994-022-06244-2) (ML 2022) [:octocat:](https://palm.seu.edu.cn/zhangml/files/PLENL.rar)

<a name="online-learning" />

### <u>Online Learning</u>

- [Online Partial Label Learning](https://link.springer.com/chapter/10.1007/978-3-030-67661-2_27) (ECML-PKDD 2020) [:octocat:](https://github.com/hbzju/OPLL)

<a name="few-shot-learning" />

### <u>Few-Shot Learning</u>

- [Robust Zero-Shot Learning with Ambiguous Labels via Visual-Semantic Alignment and Dynamic Disambiguation](https://doi.org/10.1016/j.neunet.2026.109142) (NN 2026)
- [Negatives Make a Positive: An Embarrassingly Simple Approach to Semi-Supervised Few-Shot Learning](https://doi.org/10.1109/TPAMI.2023.3333528) (TPAMI 2024)
- [Few-Shot Partial-Label Learning](https://arxiv.org/abs/2106.00984) (IJCAI 2021)

<a name="domain-adaptation" />

### <u>Domain Adaptation & Generalization</u>

- [Partial Label Causal Representation Learning for Instance-Dependent Supervision and Domain Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/35437) (AAAI 2025) [:octocat:](https://palm.seu.edu.cn/zhangml/files/CausalPLL.rar)
- [Improving Unsupervised Domain Adaptation: A Pseudo-Candidate Set Approach](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4650_ECCV_2024_paper.php) (ECCV 2024)
- [Partial Label Unsupervised Domain Adaptation with Class-Prototype Alignment](https://openreview.net/forum?id=jpq0qHggw3t) (ICLR 2023)
- [Learning from a Complementary-label Source Domain: Theory and Algorithms](https://arxiv.org/abs/2008.01454) (TNNLS 2021) [:octocat:](https://github.com/Yiyang98/BFUDA)

<a name="fed-pll" />

### <u>Federated Partial-Label Learning</u>

- [Federated Partial Label Learning with Local-Adaptive Augmentation and Regularization](https://ojs.aaai.org/index.php/AAAI/article/view/29562) (AAAI 2024)

<a name="partial-label-regression" />

### <u>Partial-Label Regression</u>

- [Learning from Interval Targets](https://openreview.net/forum?id=Q5bJZArYUZ) (NeurIPS 2025) [:octocat:](https://github.com/bloomberg/interval_targets)
- [Partial-Label Regression](https://arxiv.org/abs/2306.08968) (AAAI 2023) [:octocat:](https://palm.seu.edu.cn/zhangml/files/PLR.zip)

<a name="conformal-prediction" />

### <u>Reject Option & Conformal Prediction</u>

- [Partial-Label Learning with a Reject Option](https://arxiv.org/abs/2402.00592) (TMLR 2025) [:octocat:](https://github.com/mathefuchs/pll-with-a-reject-option)
- [Partial-Label Learning with Conformal Candidate Cleaning](https://arxiv.org/abs/2502.07661) (UAI 2025) [:octocat:](https://github.com/mathefuchs/pll-with-conformal-candidate-cleaning)
- [Conformal Prediction for Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/33853) (AAAI 2025)
- [Conformal Prediction with Partially Labeled Data](https://proceedings.mlr.press/v204/javanmardi23a.html) (COPA 2023) [:octocat:](https://github.com/pwhofman/conformal-partial-labels)

<a name="multi-dimensional" />

### <u>Multi-Dimensional Partial-Label Learning</u>

- [Learning From Multi-Dimensional Partial Labels](https://www.ijcai.org/proceedings/2020/407) (IJCAI 2020)

<a name="multi-view-learning" />

### <u>Multi-View Partial-Label Learning</u>

- [CFDM: Contrastive Fusion and Disambiguation for Multi-View Partial-Label Learning](https://doi.org/10.1609/aaai.v39i16.33869) (AAAI 2025)
- [Complementary Label Learning with Multi-View Data and a Semi-Supervised Labeling Mechanism](https://doi.org/10.1016/j.patcog.2025.111651) (PR 2025)

<a name="adversarial-training" />

### <u>Adversarial Training & Robustness</u>

- [Adversary-Aware Partial label learning with Label distillation](https://arxiv.org/abs/2304.00498) (arXiv 2023)
- [Adversarial Training with Complementary Labels: On the Benefit of Gradually Informative Attacks](https://openreview.net/forum?id=s7SukMH7ie9) (NeurIPS 2022) [:octocat:](https://github.com/RoyalSkye/ATCL)
- [Detecting the Fake Candidate Instances: Ambiguous Label Learning with Generative Adversarial Networks](https://dl.acm.org/doi/abs/10.1145/3459637.3482251) (CIKM 2021)
- [Adversarial Partial Multi-Label Learning with Label Disambiguation](https://ojs.aaai.org/index.php/AAAI/article/view/17264) (AAAI 2021)

<a name="applications" />

## <u>Applications</u>

<a name="llm" />

### <u>Large Language Models</u>

- [What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time](https://aclanthology.org/2026.acl-long.1337/) (ACL 2026) [:octocat:](https://github.com/Jasper-Yan/SCRL)

<a name="text" />

### <u>Natural Language Processing</u>

- [Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](https://aclanthology.org/2025.acl-long.139/) (ACL 2025) [:octocat:](https://github.com/MingxuanXia/CanDist)

<a name="sequence" />

### <u>Sequence</u>

- [Star Temporal Classification: Sequence Modeling with Partially Labeled Data](https://proceedings.neurips.cc/paper_files/paper/2022/hash/57587d8d6a7ede0e5302fc22d0878c53-Abstract-Conference.html) (NeurIPS 2022)

<a name="recognition" />

### <u>Computer Vision</u>

- [Partial Label Learning for Emotion Recognition from EEG](https://arxiv.org/abs/2302.13170) (TAFFC 2025) [:octocat:](https://github.com/guangyizhangbci/PLL-Emotion-EEG)
- [Partial Label Learning with Focal Loss for Sea Ice Classification Based on Ice Charts](https://arxiv.org/abs/2406.03645) (JSTARS 2024) [:octocat:](https://github.com/geohai/PLL-sea-ice-classification)
- [Navigating Real-World Partial Label Learning: Unveiling Fine-Grained Images with Attributes](https://ojs.aaai.org/index.php/AAAI/article/view/29184) (AAAI 2024)
- [A Confidence-based Partial Label Learning Model for Crowd-Annotated Named Entity Recognition](https://arxiv.org/abs/2305.12485) (ACL Findings 2023) [:octocat:](https://github.com/LemXiong/CPLL)
- [Webly-Supervised Fine-Grained Recognition with Partial Label Learning](https://www.ijcai.org/proceedings/2022/209) (IJCAI 2022) [:octocat:](https://github.com/msfuxian/WSFGPLL)

<a name="graph" />

### <u>Graph Learning</u>

- [PRISM: Partial-label Relational Inference with Spatial and Spectral Cues](https://openreview.net/forum?id=m2MeiYOJED) (ICLR 2026)
- [MATE: Masked Optimal Transport with Dynamic Selection for Partial Label Graph Learning](https://doi.org/10.1016/j.artint.2025.104396) (AIJ 2025)
- [CODE: Towards Partial Label Graph Learning via Coupled Dual Separation](https://doi.org/10.1145/3746027.3755812) (ACM MM 2025)
- [Graph Partial Label Learning with Potential Cause Discovering](https://arxiv.org/abs/2403.11449) (arXiv 2024)

<a name="retrieval" />

### <u>Cross-Modal Retrieval & Hashing</u>

- [Ambiguity-Tolerant Cross-Modal Hashing with Partial Labels](https://ojs.aaai.org/index.php/AAAI/article/view/39760) (AAAI 2026) [:octocat:](https://github.com/Rose-bud/ATCH)
- [DiCA: Disambiguated Contrastive Alignment for Cross-Modal Retrieval with Partial Labels](https://ojs.aaai.org/index.php/AAAI/article/view/34271) (AAAI 2025) [:octocat:](https://github.com/Rose-bud/DiCA)
- [Neighbor-aware Contrastive Disambiguation for Cross-Modal Hashing with Redundant Annotations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9c10afc36f1290f648cadb2cd938aa72-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/Rose-bud/NACD)
- [DiDA: Disambiguated Domain Alignment for Cross-Domain Retrieval with Partial Labels](https://ojs.aaai.org/index.php/AAAI/article/view/28150) (AAAI 2024) [:octocat:](https://github.com/lhrrrrrr/DiDA)

<a name="recommendation" />

### <u>Recommender Systems</u>

- [PARS: Partial-Label-Learning-inspired Recommender Systems](https://ojs.aaai.org/index.php/AAAI/article/view/40002) (AAAI 2026) [:octocat:](https://github.com/bjtu-lucas-nlp/PARS)

<a name="semi-supervised-learning" />

### <u>Semi-Supervised Learning</u>

- [Adaptive Fuzzy Positive Learning for Annotation-Scarce Semantic Segmentation](https://doi.org/10.1007/s11263-024-02217-1) (IJCV 2025)
- [Boosting Semi-Supervised Learning with Contrastive Complementary Labeling](https://arxiv.org/abs/2212.06643) (NN 2023) [:octocat:](https://github.com/qinyideng/ccl)

<a name="active-learning" />

### <u>Active Learning</u>

- [Next Generation Active Learning: Mixture of LLMs in the Loop](https://ojs.aaai.org/index.php/AAAI/article/view/39678) (AAAI 2026) [:octocat:](https://github.com/qijindou/MoLLIA)
- [Bidirectional Uncertainty-Based Active Learning for Open Set Annotation](https://arxiv.org/abs/2402.15198) (ECCV 2024) [:octocat:](https://github.com/chenchenzong/BUAL)
- [Exploiting counter-examples for active learning with partial labels](https://link.springer.com/article/10.1007/s10994-023-06485-9) (ML 2023) [:octocat:](https://github.com/SII-Ferenas/APLL)

<a name="noisy-label-learning" />

### <u>Noisy-Label Learning</u>

- [Bridging Generative and Discriminative Noisy-Label Learning via Direction-Agnostic EM Formulation](https://arxiv.org/abs/2308.01184) (TPAMI 2026) [:octocat:](https://github.com/lfb-1/EMNL)
- [COVER-EL: Class cOVERage-Aware Noise Correction for Crowdsourced Labels via Elimination-Based Inference](https://doi.org/10.1109/TNNLS.2026.3722653) (TNNLS 2026)
- [CLEX: Complementary Label Exchange Learning for Noisy Facial Expression Recognition](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_CLEX_Complementary_Label_Exchange_Learning_for_Noisy_Facial_Expression_Recognition_CVPR_2026_paper.html) (CVPR 2026)
- [Jo-SNC: Combating Noisy Labels Through Fostering Self- and Neighbor-Consistency](https://doi.org/10.1109/TPAMI.2025.3646737) (TPAMI 2026) [:octocat:](https://github.com/NUST-Machine-Intelligence-Laboratory/Jo-SNC)
- [Revisiting Learning with Noisy Labels: Active Forgetting and Noise Suppression](https://openaccess.thecvf.com/content/CVPR2026/html/Sheng_Revisiting_Learning_with_Noisy_Labels_Active_Forgetting_and_Noise_Suppression_CVPR_2026_paper.html) (CVPR 2026)
- [Gray Learning From Non-IID Data With Out-of-Distribution Samples](https://arxiv.org/abs/2206.09375) (TNNLS 2025) [:octocat:](https://github.com/Lawliet-zzl/GL)
- [Adaptive Integration of Partial Label Learning and Negative Learning for Enhanced Noisy Label Learning](https://arxiv.org/abs/2312.09505) (AAAI 2024) [:octocat:](https://github.com/NUST-Machine-Intelligence-Laboratory/NPN)
- [Mitigating Label Noise through Data Ambiguation](https://ojs.aaai.org/index.php/AAAI/article/view/29286) (AAAI 2024) [:octocat:](https://github.com/julilien/mitigatinglabelnoisedataambiguation)
- [NLNL: Negative Learning for Noisy Labels](https://arxiv.org/abs/1908.07387) (ICCV 2019) [:octocat:](https://github.com/ydkim1293/NLNL-Negative-Learning-for-Noisy-Labels)

<a name="TTA" />

### <u>Test-Time Adaptation</u>

- [Think Twice before Adaptation: Improving Adaptability of DeepFake Detection via Online Test-Time Adaptation](https://www.ijcai.org/proceedings/2025/854) (IJCAI 2025) [:octocat:](https://github.com/HongHanh2104/T2A-Think-Twice-Before-Adaptation)
- [Selective Label Enhancement Learning for Test-Time Adaptation](https://openreview.net/forum?id=3Z2flzXzBY) (ICLR 2025) [:octocat:](https://github.com/palm-ml/PASLE)
- [Rethinking Precision of Pseudo Label: Test-Time Adaptation via Complementary Learning](https://arxiv.org/abs/2301.06013) (PRL 2024)
- [MonoTTA: Fully Test-Time Adaptation for Monocular 3D Object Detection](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6035_ECCV_2024_paper.php) (ECCV 2024) [:octocat:](https://github.com/Hongbin98/MonoTTA)

<a name="weak-supervision-evaluation" />

### <u>Weak Supervision Evaluation</u>

- [Towards Scalable Oversight via Partitioned Human Supervision](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b6f03ce953cf7f07dafa20891d2539e3-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/R-Yin-217/Towards-Scalable-Oversight-via-Partitioned-Human-Supervision)
<!-- papers:end -->

<a name="dataset" />

## Dataset

### Tabular Dataset:

**Notice:** The following partial label learning data sets were collected and pre-processed by Prof. [Min-Ling Zhang](https://palm.seu.edu.cn/zhangml/), with courtesy and proprietary to the authors of referred literatures on them. The pre-processed data sets can be used at your own risk and for academic purpose only. More information can be found in [here](http://palm.seu.edu.cn/zhangml/).

Dataset for partial label learning:

| [FG-NET](https://palm.seu.edu.cn/zhangml/files/FG-NET.rar) | [Lost](https://palm.seu.edu.cn/zhangml/files/lost.rar) | [MSRCv2](https://palm.seu.edu.cn/zhangml/files/MSRCv2.rar) | [BirdSong](https://palm.seu.edu.cn/zhangml/files/BirdSong.rar) | [Soccer Player](https://palm.seu.edu.cn/zhangml/files/Soccer%20Player.rar) | [Yahoo! News](https://palm.seu.edu.cn/zhangml/files/Yahoo!%20News.rar) | [Mirflickr](https://palm.seu.edu.cn/zhangml/files/Mirflickr.zip) |
| ---------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

Dataset for partial multi-label learning:

| [Music_emotion](https://palm.seu.edu.cn/zhangml/files/pml_music_emotion.zip) | [Music_style](https://palm.seu.edu.cn/zhangml/files/pml_music_style.zip) | [Mirflickr](https://palm.seu.edu.cn/zhangml/files/pml_mirflickr.zip) | [YeastBP](https://palm.seu.edu.cn/zhangml/files/pml_YeastBP.zip) | [YeastCC](https://palm.seu.edu.cn/zhangml/files/pml_YeastCC.zip) | [YeastMF](https://palm.seu.edu.cn/zhangml/files/pml_YeastMF.zip) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

Data sets for multi-instance partial-label learning:

| [MNIST](https://palm.seu.edu.cn/zhangml/files/MNIST_MIPL.zip) | [FMNIST](https://palm.seu.edu.cn/zhangml/files/FMNIST_MIPL.zip) | [Newsgroups](https://palm.seu.edu.cn/zhangml/files/Newsgroups_MIPL.zip) | [Birdsong](https://palm.seu.edu.cn/zhangml/files/Birdsong_MIPL.zip) | [SIVAL](https://palm.seu.edu.cn/zhangml/files/SIVAL_MIPL.zip) | [CRC-Row](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-Row.zip) | [CRC-SBN](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-SBN.zip) | [CRC-KMeansSeg](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-KMeansSeg.zip) | [CRC-SIFT](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-SIFT.zip) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

### Image Dataset:

- Dataset for complementary-label learning: [CLImage dataset](https://github.com/ntucllab/CLImage_Dataset)
- Dataset for partial-label learning: [PLCIFAR10 dataset](https://github.com/wwangwitsel/PLCIFAR10) (human-annotated partial labels, from the PLENCH benchmark)
