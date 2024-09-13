# Advances in Partial/Complementary Label Learning 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/> <img src="https://img.shields.io/badge/Number%20of%20Items-168-FF6F00" alt="PaperNum"/> ![Stars](https://img.shields.io/github/stars/wu-dd/Advances-in-Partial-and-Complementary-Label-Learning?color=yellow&label=Stars) ![Forks](https://img.shields.io/github/forks/wu-dd/Advances-in-Partial-and-Complementary-Label-Learning?color=green&label=Forks)





**Advances in Partial/Complementary Label Learning** provides the most advanced and detailed information on partial/complementary label learning field.

**Partial/complementary label learning** is an emerging framework in weakly supervised machine learning with broad application prospects. It handles the case in which each training example corresponds to a candidate label set and only one label concealed in the set is the ground-truth label.

**This project is curated and maintained by [Dong-Dong Wu](https://wu-dd.github.io/)**. I will do my best to keep the project up to date. If you have any suggestions or are interested in being contributors, feel free to drop me an email.



#### [How to submit a pull request?](./CONTRIBUTING.md)

+ :globe_with_meridians: Project Page
+ :octocat: Code
+ :book: `bibtex`

## Latest Updates
+ [2024/09/13] A major overhaul of the original github repository. 

## Contents
- [Main](#main)
  - [Early Work](#early-work)
  - [Theory-Based](#theory-based)
  - [Transition Matrix-Based](#transition-matrix-based)
  - [Understanding](#understanding)
  - [Better Optimization](#better-optimization)
  - [Partial Multi-Label Learning](#partial-multi-label-learning)
  - [Noisy Partial Label Learning](#noisy-partial-label-learning)
  - [Semi-Supervised Partial Label Learning](#semi-supervised-partial-label-learning)
  - [Multi-Instance Partial Label Learning](#multi-instance-learning)
  - [Imbalanced Partial Label Problem](#class-imbalance-problem)
  - [Out-of-distributrion Partial Label Learning](#partial-label-regression)
  - [Partial Label Regression](#partial-label-regression)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Multi-Complementary Label Learning](#multi-complementary-label-learning)
  - [Multi-View Learning](#multi-view-learning)
  - [Adversarial Training](#adversarial-training)
  - [Negative Learning](#negative-learning)
  - [Incremental Learning](#incremental-learning)
  - [Online Learning](#online-learning)
  - [Conformal Prediction](#conformal-prediction)
  - [Few-Shot Learning](#few-shot-learning)
  - [Open-Set Problem](#open-set-problem)
  - [Data Augmentation](#data-augmentation)
  - [Multi-Dimensional](#multi-dimensional)
  - [Domain Adaptation](#domain-adaptation)
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

<a name="main" />

## <u>Main</u>

- [Learning from Complementary Labels](https://arxiv.org/abs/1705.07541) (NeurIPS 2018) [:octocat:](https://github.com/takashiishida/comp)

<a name="early-work" />

### <u>Early Work</u>
+ To be continue.

<a name="theory-based" />

### <u>Theory-based</u>

- [Complementary-Label Learning for Arbitrary Losses and Models](https://arxiv.org/abs/1810.04327) (ICML 2019)
- [Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels](https://arxiv.org/abs/2007.02235) (ICML 2020)
- [Leveraged Weighted Loss for Partial Label Learning](https://arxiv.org/abs/2106.05731) (ICML 2021)
- [Unbiased Risk Estimator to Multi-Labeled Complementary Label Learning](https://www.ijcai.org/proceedings/2023/415) (IJCAI 2023)
- [Learning with Complementary Labels Revisited: The Selected-Completely-at-Random Setting Is More Practical](https://arxiv.org/abs/2311.15502) (ICML 2024)
- [Towards Unbiased Exploration in Partial Label Learning](https://arxiv.org/abs/2307.00465)

<a name="transition-matrix-based" />

### <u>Transition-Matrix-Based</u>

- [Learning with Biased Complementary Labels](https://arxiv.org/abs/1711.09535)  (ECCV 2018)

<a name="understanding" />

### <u>Understanding</u>

- [Bridging Ordinary-Label Learning and Complementary-Label Learning](https://proceedings.mlr.press/v129/katsura20a.html) (ACML 2020)
- [On the Power of Deep but Naive Partial Label Learning](https://arxiv.org/abs/2010.11600) (ICASSP 2021)
- [Learning from a Complementary-label Source Domain: Theory and Algorithms](https://arxiv.org/abs/2008.01454) (TNNLS 2021)
- [A Unifying Probabilistic Framework for Partially Labeled Data Learning](https://ieeexplore.ieee.org/document/9983986) (TPAMI 2023)
- [Candidate Label Set Pruning: A Data-centric Perspective for Deep Partial-label Learning](https://openreview.net/forum?id=Fk5IzauJ7F) (ICLR 2024)
- [Understanding Self-Distillation and Partial Label Learning in Multi-Class Classification with Label Noise](https://arxiv.org/abs/2402.10482)

<a name="better-optimization" />

### <u>Better Optimization</u>

- [GM-PLL: Graph Matching based Partial Label Learning](https://arxiv.org/pdf/1901.03073) (TKDE 2019)
- [Partial Label Learning via Label Enhancement](https://ojs.aaai.org/index.php/AAAI/article/view/4497) (AAAI 2019)
- [Partial Label Learning with Self-Guided Retraining](https://arxiv.org/abs/1902.03045) (AAAI 2019)
- [Partial Label Learning by Semantic Difference Maximization](https://www.ijcai.org/proceedings/2019/318) (IJCAI 2019)
- [Partial Label Learning with Unlabeled Data](https://www.ijcai.org/proceedings/2019/521) (IJCAI 2019)
- [Adaptive Graph Guided Disambiguation for Partial Label Learning](https://dl.acm.org/doi/10.1145/3292500.3330840) (KDD 2019)
- [A Self-Paced Regularization Framework for Partial-Label Learning](https://ieeexplore.ieee.org/document/9094702) (TYCB 2020)
- [Large Margin Partial Label Machine](https://ieeexplore.ieee.org/document/8826247) (TNNLS 2020)
- [Learning with Noisy Partial Labels by Simultaneously Leveraging Global and Local Consistencies](https://dl.acm.org/doi/10.1145/3340531.3411885) (CIKM 2020)
- [Network Cooperation with Progressive Disambiguation for Partial Label Learning](https://arxiv.org/abs/2002.11919) (ECML-PKDD 2020)
- [Deep Discriminative CNN with Temporal Ensembling for Ambiguously-Labeled Image Classification](https://ojs.aaai.org/index.php/AAAI/article/view/6959) (AAAI 2020)
- [Generative-Discriminative Complementary Learning](https://arxiv.org/abs/1904.01612) (AAAI 2020)
- [Partial Label Learning with Batch Label Correction](https://ojs.aaai.org/index.php/AAAI/article/view/6132) (AAAI 2020)
- [Progressive Identification of True Labels for Partial-Label Learning](https://arxiv.org/abs/2002.08053) (ICML 2020)
- [Provably Consistent Partial-Label Learning](https://arxiv.org/abs/2007.08929) (NeurIPS 2020)
- [Generalized Large Margin -NN for Partial Label Learning](https://ieeexplore.ieee.org/document/9529072) (TMM2021)
- [Adaptive Graph Guided Disambiguation for Partial Label Learning](https://ieeexplore.ieee.org/document/9573413) (TPAMI 2022)
- [Discriminative Metric Learning for Partial Label Learning](https://ieeexplore.ieee.org/document/9585342) (TNNLS 2021)
- [Top-k Partial Label Machine](https://ieeexplore.ieee.org/document/9447152) (TNNLS 2021)
- [Detecting the Fake Candidate Instances: Ambiguous Label Learning with Generative Adversarial Networks](https://dl.acm.org/doi/abs/10.1145/3459637.3482251) (CIKM 2021)
- [Discriminative Complementary-Label Learning with Weighted Loss](https://proceedings.mlr.press/v139/gao21d.html) (ICML 2021)
- [Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2110.12911) (NeurIPS 2021)
- [Learning with Proper Partial Labels](https://arxiv.org/abs/2112.12303) (NearoComputing 2022)
- [Biased Complementary-Label Learning Without True Labels](https://ieeexplore.ieee.org/document/9836971) (TNNLS 2022)
- [Exploiting Class Activation Value for Partial-Label Learning](https://openreview.net/forum?id=qqdXHUGec9h) (ICLR 2022)
- [PiCO: Contrastive Label Disambiguation for Partial Label Learning](https://openreview.net/forum?id=EhYjZy6e1gJ) (ICLR 2022)
- [Deep Graph Matching for Partial Label Learning](https://www.ijcai.org/proceedings/2022/459) (IJCAI 2022)
- [Exploring Binary Classification Hidden within Partial Label Learning](https://www.ijcai.org/proceedings/2022/456) (IJCAI 2022)
- [Partial Label Learning via Label Influence Function](https://proceedings.mlr.press/v162/gong22c.html) (ICML 2022)
- [Revisiting Consistency Regularization for Deep Partial Label Learning](https://proceedings.mlr.press/v162/wu22l.html) (ICML 2022)
- [Partial Label Learning with Semantic Label Representations ](https://dl.acm.org/doi/abs/10.1145/3534678.3539434) (KDD 2022)
- [GraphDPI: Partial label disambiguation by graph representation learning via mutual information maximization](https://www.sciencedirect.com/science/article/abs/pii/S0031320322006136) (PR 2023)
- [Variational Label Enhancement](https://ieeexplore.ieee.org/document/9875104) (TPAMI 2023)
- [CMW-Net: Learning a Class-Aware Sample Weighting Mapping for Robust Deep Learning](https://arxiv.org/abs/2202.05613) (TPAMI 2023)
- [Reduction from Complementary-Label Learning to Probability Estimates](https://arxiv.org/abs/2209.09500) (PAKDD 2023)
- [Decompositional Generation Process for Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2204.03845) (ICLR 2023)
- [Mutual Partial Label Learning with Competitive Label Noise](https://openreview.net/forum?id=EUrxG8IBCrC) (ICLR 2023)
- [Can Label-Specific Features Help Partial-Label Learning? ](https://ojs.aaai.org/index.php/AAAI/article/view/25904) (AAAI 2023)
- [Learning with Partial Labels from Semi-supervised Perspective](https://arxiv.org/abs/2211.13655) (AAAI 2023)
- [Consistent Complementary-Label Learning via Order-Preserving Losses](https://proceedings.mlr.press/v206/liu23g.html) (ICAIS 2023)
- [Complementary Classifier Induced Partial Label Learning](https://arxiv.org/abs/2305.09897) (KDD 2023)
- [Towards Effective Visual Representations for Partial-Label Learning](https://arxiv.org/abs/2305.06080) (CVPR 2023)
- [Candidate-aware Selective Disambiguation Based On Normalized Entropy for Instance-dependent Partial-label Learning](https://ieeexplore.ieee.org/document/10376678) (ICCV 2023)
- [Partial Label Learning with Dissimilarity Propagation guided Candidate Label Shrinkage](https://papers.nips.cc/paper_files/paper/2023/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html) (NeurIPS 2023)
- [Learning From Biased Soft Labels](https://arxiv.org/abs/2302.08155) (NeurIPS 2023)
- [Self-distillation and self-supervision for partial label learning](https://www.sciencedirect.com/science/article/pii/S0031320323007136) (PR 2023)
- [Partial Label Learning with a Partner](https://ojs.aaai.org/index.php/AAAI/article/view/29424) (AAAI 2024)
- [Distilling Reliable Knowledge for Instance-Dependent Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29519) (AAAI 2024)
- [Disentangled Partial Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/28976) (AAAI 2024)
- [CroSel: Cross Selection of Confident Pseudo Labels for Partial-Label Learning](https://arxiv.org/abs/2303.10365) (CVPR 2024)
- [A General Framework for Learning from Weak Supervision](https://arxiv.org/abs/2402.01922) (ICML 2024)
- [Does Label Smoothing Help Deep Partial Label Learning?](https://openreview.net/forum?id=drjjxmi2Ha) (ICML 2024)
- [Label Dropout: Improved Deep Learning Echocardiography Segmentation Using Multiple Datasets With Domain Shift and Partial Labelling](https://arxiv.org/abs/2403.07818) (SCIS 2024)
- [Meta Objective Guided Disambiguation for Partial Label Learning](https://arxiv.org/abs/2208.12459)
- [Adversary-Aware Partial label learning with Label distillation](https://arxiv.org/abs/2304.00498)
- [Solving Partial Label Learning Problem with Multi-Agent Reinforcement Learning](https://openreview.net/forum?id=BNsuf5g-JRd)
- [Learning from Stochastic Labels](https://arxiv.org/abs/2302.00299)
- [Deep Duplex Learning for Weak Supervision](https://openreview.net/forum?id=SeZ5ONageGl)
- [Imprecise Label Learning: A Unified Framework for Learning with Various Imprecise Label Configurations](https://arxiv.org/abs/2305.12715)
- [Enhancing Label Sharing Efficiency in Complementary-Label Learning with Label Augmentation](https://arxiv.org/abs/2305.08344)
- [Appeal: Allow Mislabeled Samples the Chance to be Rectified in Partial Label Learning](https://arxiv.org/abs/2312.11034)
- [Graph Partial Label Learning with Potential Cause Discovering](https://arxiv.org/abs/2403.11449) (2024)

<a name="partial-multi-label-learning" />

### <u>Partial-Multi Label Learning</u>

- [Learning a Deep ConvNet for Multi-label Classification with Partial Labels](https://arxiv.org/abs/1902.09720) (CVPR 2019)
- [Multi-View Partial Multi-Label Learning with Graph-Based Disambiguation](https://ojs.aaai.org/index.php/AAAI/article/view/5761) (AAAI 2020)
- [Partial Multi-Label Learning via Multi-Subspace Representation](https://www.ijcai.org/proceedings/2020/362) (IJCAI 2020)
- [Feature-Induced Manifold Disambiguation for Multi-View Partial Multi-label Learning](https://dl.acm.org/doi/10.1145/3394486.3403098) (KDD 2020)
- [Prior Knowledge Regularized Self-Representation Model for Partial Multilabel Learning](https://ieeexplore.ieee.org/document/9533180) (TYCB 2021)
- [Global-Local Label Correlation for Partial Multi-Label Learning](https://ieeexplore.ieee.org/document/9343691) (TMM 2021)
- [Progressive Enhancement of Label Distributions for Partial Multilabel Learning](https://ieeexplore.ieee.org/document/9615493) (TNNLS 2021)
- [Partial Multi-Label Learning With Noisy Label Identification](https://ieeexplore.ieee.org/document/9354590) (TPAMI 2021)
- [Partial Multi-Label Learning via Credible Label Elicitation](https://ieeexplore.ieee.org/document/9057438) (TPAMI 2021)
- [Adversarial Partial Multi-Label Learning](https://arxiv.org/abs/1909.06717) (AAAI 2021)
- [Learning from Complementary Labels via Partial-Output Consistency Regularization](https://www.ijcai.org/proceedings/2021/423) (IJCAI 2021)
- [Partial Multi-Label Learning with Meta Disambiguation](https://dl.acm.org/doi/abs/10.1145/3447548.3467259) (KDD 2021)
- [Understanding Partial Multi-Label Learning via Mutual Information](https://proceedings.neurips.cc/paper/2021/hash/217c0e01c1828e7279051f1b6675745d-Abstract.html) (NeurIPS 2021)
- [Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels](https://arxiv.org/abs/2203.02172) (AAAI 2022)
- [Structured Semantic Transfer for Multi-Label Recognition with Partial Labels)](https://arxiv.org/abs/2112.10941) (AAAI 2022)
- [Boosting Multi-Label Image Classification with Complementary Parallel Self-Distillation](https://arxiv.org/abs/2205.10986) (IJCAI 2022)
- [Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning](https://www.ijcai.org/proceedings/2022/502) (IJCAI 2022)
- [Multi-label Classification with Partial Annotations using Class-aware Selective Loss](https://arxiv.org/abs/2110.10955) (CVPR 2022)
- [Deep Double Incomplete Multi-View Multi-Label Learning With Incomplete Labels and Missing Views](https://ieeexplore.ieee.org/document/10086538) (TNNLS 2023)
- [Towards Enabling Binary Decomposition for Partial Multi-Label Learning](https://ieeexplore.ieee.org/document/10168295) (TPAMI 2023)
- [Deep Partial Multi-Label Learning with Graph Disambiguation](https://arxiv.org/abs/2305.05882) (IJCAI 2023)
- [Learning in Imperfect Environment: Multi-Label Classification with Long-Tailed Distribution and Partial Labels](https://arxiv.org/abs/2304.10539) (ICCV 2023)
- [Partial Multi-Label Learning with Probabilistic Graphical Disambiguation](https://papers.nips.cc/paper_files/paper/2023/hash/04e05ba5cbc36044f6499d1edf15247e-Abstract-Conference.html) (NeurIPS 2023)
- [ProPML: Probability Partial Multi-label Learning](https://arxiv.org/abs/2403.07603) (DSAA 2023)
- [A Deep Model for Partial Multi-Label Image Classification with Curriculum Based Disambiguation](https://arxiv.org/abs/2207.02410) (ML 2024)
- [Partial Multi-View Multi-Label Classification via Semantic Invariance Learning and PrototypeModeling](https://icml.cc/virtual/2024/poster/34972) (ICML 2024)
- [Reliable Representations Learning for Incomplete Multi-View Partial Multi-Label Classification](https://arxiv.org/abs/2303.17117) (2024)
- [PLMCL: Partial-Label Momentum Curriculum Learning for Multi-Label Image Classification](https://arxiv.org/abs/2208.09999) (2024)
- [Combining Supervised Learning and Reinforcement Learning for Multi-Label Classification Tasks with Partial Labels](https://arxiv.org/abs/2406.16293) (2024)

<a name="noisy-partial-label-learning" />

### <u>Noisy Partial Label Learning</u>

- [PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning](https://arxiv.org/abs/2201.08984) (TPAMI 2023)
- [On the Robustness of Average Losses for Partial-Label Learning](https://arxiv.org/abs/2106.06152) (TPAMI 2023)
- [FREDIS: A Fusion Framework of Refinement and Disambiguation for Unreliable Partial Label Learning](https://proceedings.mlr.press/v202/qiao23b.html) (ICML 2023)
- [Progressive Purification for Instance-Dependent Partial Label Learning](https://arxiv.org/abs/2206.00830) (ICML 2023)
- [Unreliable Partial Label Learning with Recursive Separation](https://arxiv.org/abs/2302.09891) (IJCAI 2023)
- [ALIM: Adjusting Label Importance Mechanism for Noisy Partial Label Learning](https://arxiv.org/abs/2301.12077) (NeurIPS 2023)
- [IRNet: Iterative Refinement Network for Noisy Partial Label Learning](https://arxiv.org/abs/2211.04774)
- [Robust Representation Learning for Unreliable Partial Label Learning](https://arxiv.org/abs/2308.16718)
- [Pseudo-labelling meets Label Smoothing for Noisy Partial Label Learning](https://arxiv.org/abs/2402.04835)

<a name="semi-supervised-partial-label-learning" />

### <u>Semi-Supervised Partial Label Learning</u>

- [Semi-Supervised Partial Label Learning via Confidence-Rated Margin Maximization](https://proceedings.neurips.cc/paper/2020/hash/4dea382d82666332fb564f2e711cbc71-Abstract.html) (NeurIPS 2020)
- [Exploiting Unlabeled Data via Partial Label Assignment for Multi-Class Semi-Supervised Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17310) (AAAI 2021)
- [Distributed Semisupervised Partial Label Learning Over Networks](https://ieeexplore.ieee.org/document/9699063) (AI 2022)
- [Learning with Partial-Label and Unlabeled Data: A Uniform Treatment for Supervision Redundancy and Insufficiency](https://proceedings.mlr.press/v235/liu24ar.html) (ICML 2024)

<a name="multi-instance-learning" />

### <u>Multi-Instance Partial Label Learning</u>

- [Multi-Instance Partial-Label Learning: Towards Exploiting Dual Inexact Supervision](https://arxiv.org/abs/2212.08997) (SCIS 2023)
- [Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning](https://arxiv.org/abs/2305.16912) (NeurIPS 2023)
- [Exploiting Conjugate Label Information for Multi-Instance Partial-Label Learning](https://arxiv.org/abs/2408.14369) (2024)
- [On Characterizing and Mitigating Imbalances in Multi-Instance Partial Label Learning](https://arxiv.org/abs/2407.10000) (2024)

<a name="class-imbalance-problem" />

### <u>Imblanced Partial Label Learning</u>

- [Towards Mitigating the Class-Imbalance Problem for Partial Label Learning](https://dl.acm.org/doi/10.1145/3219819.3220008) (KDD 2018) [:octocat:](https://github.com/seu71wj/CIMAP)
- [A Partial Label Metric Learning Algorithm for Class Imbalanced Data](https://proceedings.mlr.press/v157/liu21f.html) (ACML 2021)
- [SoLar: Sinkhorn Label Refinery for Imbalanced Partial-Label Learning](https://arxiv.org/abs/2209.10365) (NeurIPS 2022)
- [Class-Imbalanced Complementary-Label Learning via Weighted Loss](https://arxiv.org/abs/2209.14189) (NN 2023)
- [Partial label learning: Taxonomy, analysis and outlook](https://www.sciencedirect.com/science/article/abs/pii/S0893608023000825) (NN 2023)
- [Long-Tailed Partial Label Learning via Dynamic Rebalancing](https://arxiv.org/abs/2302.05080) (ICLR 2023)
- [Long-Tailed Partial Label Learning by Head Classifier and Tail Classifier Cooperation](https://ojs.aaai.org/index.php/AAAI/article/view/29182) (AAAI 2024)
- [Pseudo Labels Regularization for Imbalanced Partial-Label Learning](https://arxiv.org/abs/2303.03946)

<a name="ood-pll" />

### Out-of-distribution Partial Label Learning

- [Out-of-distribution Partial Label Learning](https://arxiv.org/abs/2403.06681) (2024)

<a name="partial-label-regression" />

### <u>Partial Label Regression</u>

- [Partial-Label Regression](https://arxiv.org/abs/2306.08968) (AAAI 2023)
- [Partial-Label Learning with a Reject Option](https://arxiv.org/abs/2402.00592)

<a name="dimensionality-reduction" />

### <u>Dimensionality Reduction</u>

- [Partial Label Dimensionality Reduction via Confidence-Based Dependence Maximization](https://dl.acm.org/doi/abs/10.1145/3447548.3467313) (KDD 2021)
- [Disambiguation Enabled Linear Discriminant Analysis for Partial Label Dimensionality Reduction](https://dl.acm.org/doi/abs/10.1145/3494565) (TKDD 2022)
- [Submodular Feature Selection for Partial Label Learning](https://dl.acm.org/doi/abs/10.1145/3534678.3539292) (KDD 2022)
- [Dimensionality Reduction for Partial Label Learning: A Unified and Adaptive Approach](https://www.computer.org/csdl/journal/tk/2024/08/10440495/1UGSdp8q3Wo) (TKDE 2024)

<a name="multi-complementary-label-learning" />

### <u>Multi-Complementary Label Learning</u>

- [Learning with Multiple Complementary Labels](https://arxiv.org/abs/1912.12927) (ICML 2020)
- [Multi-Complementary and Unlabeled Learning for Arbitrary Losses and Models](https://arxiv.org/abs/2001.04243) (PR 2022)

<a name="multi-view-learning" />

### <u>Multi-View Learning</u>

- [Deep Partial Multi-View Learning](https://ieeexplore.ieee.org/document/9258396) (TPAMI 2022)

<a name="adversarial-training" />

### <u>Adversarial Training</u>

- [Adversarial Training with Complementary Labels: On the Benefit of Gradually Informative Attacks](https://openreview.net/forum?id=s7SukMH7ie9) (NeurIPS 2022)

<a name="negative-learning" />

### <u>Negative Learning</u>

- [NLNL: Negative Learning for Noisy Labels](https://arxiv.org/abs/1908.07387) (ICCV 2019)

<a name="incremental-learning" />

### <u>Incremental Learning</u>

- [Partial label learning with emerging new labels](https://link.springer.com/article/10.1007/s10994-022-06244-2) (ML 2022)
- [Complementary Labels Learning with Augmented Classes](https://arxiv.org/abs/2211.10701) （2022）

<a name="online-learning" />

### <u>Online Learning</u>

- [Online Partial Label Learning](https://dl.acm.org/doi/abs/10.1007/978-3-030-67661-2_27) (ECML-PKDD 2020)

<a name="conformal-prediction" />

### <u>Conformal Prediction</u>

- [Conformal Prediction with Partially Labeled Data](https://arxiv.org/abs/2306.01191) (SCPPA 2023)

<a name="few-shot-learning" />

### <u>Few-Shot Learning</u>

- [Few-Shot Partial-Label Learning](https://arxiv.org/abs/2106.00984) (IJCAI 2021)

<a name="open-set-problem" />

### <u>Open-set Problem</u>

- [Partial-label Learning with Mixed Closed-set and Open-set Out-of-candidate Examples](https://arxiv.org/abs/2307.00553) (KDD 2023)

<a name="data-augmentation" />

### <u>Data Augmentation</u>

- [Partial Label Learning with Discrimination Augmentation](https://dl.acm.org/doi/abs/10.1145/3534678.3539363) (KDD 2022)

<a name="multi-dimensional" />

### <u>Multi-Dimensional</u>

- [Learning From Multi-Dimensional Partial Labels](https://www.ijcai.org/proceedings/2020/407) (IJCAI 2020)

<a name="domain-adaptation" />

### <u>Domain Adaptation</u>

- [Partial Label Unsupervised Domain Adaptation with Class-Prototype Alignment](https://openreview.net/forum?id=jpq0qHggw3t) (ICLR 2023)



<a name="applications" />

## <u>Applications</u>

<a name="audio" />

### <u>Audio</u>

- [Semi-Supervised Audio Classification with Partially Labeled Data](https://arxiv.org/abs/2111.12761)

<a name="text"/>

### <u>Text</u>

- [Complementary Auxiliary Classifiers for Label-Conditional Text Generation](https://ojs.aaai.org/index.php/AAAI/article/view/6346) (AAAI 2020)

<a name="sequence" />

### <u>Sequence</u>

- [Star Temporal Classification: Sequence Classification with Partially Labeled Data](https://arxiv.org/abs/2201.12208)

<a name="recognition"/>

### <u>Recognition</u>

- [Webly-Supervised Fine-Grained Recognition with Partial Label Learning](https://www.ijcai.org/proceedings/2022/209) (IJCAI 2022)
- [Rethinking the Learning Paradigm for Dynamic Facial Expression Recognition](https://ieeexplore.ieee.org/document/10204167) (CVPR 2023)
- [Partial Label Learning with Focal Loss for Sea Ice Classification Based on Ice Charts](https://arxiv.org/abs/2406.03645) (AEORS 2023)
- [Partial Label Learning for Emotion Recognition from EEG](https://arxiv.org/abs/2302.13170)
- [A Confidence-based Partial Label Learning Model for Crowd-Annotated Named Entity Recognition](https://arxiv.org/abs/2305.12485)

<a name="object-localization"/>

### <u>Object Localization</u>

- [Adversarial Complementary Learning for Weakly Supervised Object Localization](https://arxiv.org/abs/1804.06962) (CVPR 2018) [:octocat:](https://github.com/halbielee/ACoL_pytorch)
- [Learning to Detect Instance-level Salient Objects Using Complementary Image Labels](https://arxiv.org/abs/2111.10137) (2021)

<a name="map-reconstruction"/>

### <u>Map Reconstruction</u>

- [Deep Learning with Partially Labeled Data for Radio Map Reconstruction](https://arxiv.org/abs/2306.05294) 

<a name="semi-supervised-learning"/>

### <u>Semi-supervised Learning</u>

- [Boosting Semi-Supervised Learning with Contrastive Complementary Labeling](https://arxiv.org/abs/2212.06643) (NN 2023)
- [Controller-Guided Partial Label Consistency Regularization with Unlabeled Data](https://arxiv.org/abs/2210.11194) (2024)
- [Semi-supervised Contrastive Learning Using Partial Label Information](https://arxiv.org/abs/2003.07921) (2024)

<a name="active-learning"/>

### <u>Active Learning</u>

- [Exploiting counter-examples for active learning with partial labels](https://link.springer.com/article/10.1007/s10994-023-06485-9) (ML 2023)
- [Active Learning with Partial Labels](https://openreview.net/forum?id=1VuBdlNBuR)

<a name="noisy-label-learning"/>

### <u>Noisy Label Learning</u>

- [Learning from Noisy Labels with Complementary Loss Functions](https://ojs.aaai.org/index.php/AAAI/article/view/17213) (AAAI 2021)
- [Adaptive Integration of Partial Label Learning and Negative Learning for Enhanced Noisy Label Learning](https://arxiv.org/abs/2312.09505) (AAAI 2024)
- [Partial Label Supervision for Agnostic Generative Noisy Label Learning](https://arxiv.org/abs/2308.01184)

<a name="TTA"/>

### <u>Test-Time Adaptation</u>

[Rethinking Precision of Pseudo Label: Test-Time Adaptation via Complementary Learning](https://arxiv.org/abs/2301.06013) (2023)





## Dataset

### Tabular Dataset:

**Notice:** The following partial label learning data sets were collected and pre-processed by Prof. [Min-Ling Zhang](https://palm.seu.edu.cn/zhangml/), with courtesy and proprietary to the authors of referred literatures on them. The pre-processed data sets can be used at your own risk and for academic purpose only. More information can be found in [here](http://palm.seu.edu.cn/zhangml/).

Dataset for partial label learning

| [FG-NET](https://palm.seu.edu.cn/zhangml/files/FG-NET.rar) | [Lost](https://palm.seu.edu.cn/zhangml/files/lost.rar) | [MSRCv2](https://palm.seu.edu.cn/zhangml/files/MSRCv2.rar) | [BirdSong](https://palm.seu.edu.cn/zhangml/files/BirdSong.rar) | [Soccer Player](https://palm.seu.edu.cn/zhangml/files/SoccerPlayer.rar) | [Yahoo! News](https://palm.seu.edu.cn/zhangml/files/Yahoo!News.rar) | [Mirflickr](https://palm.seu.edu.cn/zhangml/files/Mirflickr.zip) |
| ---------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

Dataset for partial multi-label learning

| [Music_emotion](https://palm.seu.edu.cn/zhangml/files/pml_music_emotion.zip) | [Music_style](https://palm.seu.edu.cn/zhangml/files/pml_music_style.zip) | [Mirflickr](https://palm.seu.edu.cn/zhangml/files/pml_mirflickr.zip) | [YeastBP](https://palm.seu.edu.cn/zhangml/files/pml_YeastBP.zip) | [YeastCC](https://palm.seu.edu.cn/zhangml/files/pml_YeastCC.zip) | [YeastMF](https://palm.seu.edu.cn/zhangml/files/pml_YeastMF.zip) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

Data sets for multi-instance partial-label learning:

| [MNIST](https://palm.seu.edu.cn/zhangml/files/MNIST_MIPL.zip) | [FMNIST](https://palm.seu.edu.cn/zhangml/files/FMNIST_MIPL.zip) | [Newsgroups](https://palm.seu.edu.cn/zhangml/files/Newsgroups_MIPL.zip) | [Birdsong](https://palm.seu.edu.cn/zhangml/files/Birdsong_MIPL.zip) | [SIVAL](https://palm.seu.edu.cn/zhangml/files/SIVAL_MIPL.zip) | [CRC-Row](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-Row.zip) | [CRC-SBN](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-SBN.zip) | [CRC-KMeansSeg](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-KMeansSeg.zip) | [CRC-SIFT](https://palm.seu.edu.cn/zhangml/files/CRC-MIPL-SIFT.zip) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

### Image Dataset:

- [CLImage: Human-Annotated Datasets for Complementary-Label Learning](https://arxiv.org/abs/2305.08295) (2024)



## Learderboard

To be continue.



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
