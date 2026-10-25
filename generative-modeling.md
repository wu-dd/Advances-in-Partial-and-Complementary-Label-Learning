# Generative Modeling with Imperfect Information

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/> <img src="https://img.shields.io/badge/Number%20of%20Items-175-FF6F00" alt="PaperNum"/>

A curated list of recent papers (2024 – now) on **generative modeling with imperfect information**, with a focus on **image generation**.

**Scope.** A paper is included when the information used to *train or fine-tune* the generative model (diffusion, flow matching, consistency, GAN, VAE, autoregressive) is imperfect: the training images are corrupted, noisy or incomplete, the labels / captions / conditions are noisy or weak, the data are contaminated or biased, or the feedback used for alignment is noisy. Pure inverse-problem methods that apply a prior trained on clean data to degraded test inputs are out of scope. Much of the corrupted-data line builds on [Ambient Diffusion](https://proceedings.neurips.cc/paper_files/paper/2023/hash/012af729c5d14d279581fc8a5db975a1-Abstract-Conference.html) (NeurIPS 2023) [:octocat:](https://github.com/giannisdaras/ambient-diffusion).

Papers are from top conferences and journals; a few recent, notable arXiv preprints are included and marked `arXiv`. Within each section, the newest papers come first.

This list is part of [Advances in Learning with Imperfect Information](./README.md), curated and maintained by [Dong-Dong Wu](https://wu-dd.github.io/). You can also [browse, search and filter it online](https://wu-dd.github.io/Advances-in-Learning-with-Imperfect-Information/#gen). Suggestions are welcome by email.

+ :octocat: Code (106 papers have official code)

## Latest Updates
+ [2026/10/25] First release.

<!-- papers:start -->
## Contents
- [Imperfect Training Data](#imperfect-training-data)
  - [Learning from Corrupted / Noisy Observations](#corrupted-observations)
  - [Medical Imaging](#medical-imaging)
  - [Scientific Imaging](#scientific-imaging)
  - [Missing / Incomplete Data](#missing-data)
  - [Contaminated, Outlier & Harmful Data](#contaminated-data)
  - [Imbalanced, Long-Tailed & Biased Data](#imbalanced-biased)
  - [Limited Data](#limited-data)
  - [Synthetic Data & Model Collapse](#synthetic-data)
- [Imperfect Labels & Conditions](#imperfect-labels-conditions)
  - [Noisy Labels](#noisy-labels)
  - [Noisy Captions & Text-Image Misalignment](#noisy-captions)
  - [Weak Supervision (Partial / Complementary / PU / Semi-Supervised)](#weak-supervision)
  - [Imperfect Conditions](#imperfect-conditions)
- [Imperfect Feedback & Rewards](#imperfect-feedback-rewards)
  - [Noisy Preferences](#noisy-preferences)
  - [Imperfect Reward Models](#imperfect-rewards)
  - [Weak / Sparse Feedback](#weak-feedback)
- [Theory, Privacy & Others](#theory-privacy-others)
  - [Theory & Analysis](#theory)
  - [Memorization, Privacy & Copyright](#memorization-privacy)
  - [Others](#other)

<a name="imperfect-training-data" />

## <u>Imperfect Training Data</u>

<a name="corrupted-observations" />

### <u>Learning from Corrupted / Noisy Observations</u>

- [Generative Modeling from Black-box Corruptions via Self-Consistent Stochastic Interpolants](https://proceedings.iclr.cc/paper_files/paper/2026/hash/aa482e630a0fe3153b7ab3d09950c71f-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/modichirag/SCSI)
- [SFBD-OMNI: Bridge Models for Lossy Measurement Restoration with Limited Clean Samples](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bbe0acd1ad5cf41bb87a484992a70cbd-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/watml/SFBD-omni)
- [Normalization-equivariant Diffusion Models: Learning Posterior Samplers From Noisy And Partial Measurements](https://openreview.net/forum?id=Xwxz41eYE6) (ICML 2026) [:octocat:](https://github.com/blevac/NormEQ-Diff)
- [Noise-Robust Conditional Flow Matching: Generating Clean Samples from Noisy Datasets](https://arxiv.org/abs/2608.00064) (arXiv 2026)
- [SFBD Flow: A Continuous-Optimization Framework for Training Diffusion Models with Noisy Samples](https://openreview.net/forum?id=bm9E45jCdu) (AISTATS 2026) [:octocat:](https://github.com/watml/SFBD-flow)
- [Measurement Score-Based Diffusion Model](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9bcc5e97c6343b52dc4ff7b034cc640c-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/wustl-cig/MSM)
- [Score Distillation Beyond Acceleration: Generative Modeling from Corrupted Data](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d5f893137b4e8a8ebc33cc8c2eee9e46-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/TianyuCodings/RSD)
- [Optimizing Diffusion Priors in Image Reconstruction from a Single Observation](https://arxiv.org/abs/2604.21066) (arXiv 2026)
- [Ambient Dataloops: Generative Models for Dataset Refinement](https://openreview.net/forum?id=Li5ki5Dopo) (ICML 2026) [:octocat:](https://github.com/adrianrm99/ambient_dataloops)
- [Riemannian AmbientFlow: Towards Simultaneous Manifold Learning and Generative Modeling from Corrupted Data](https://arxiv.org/abs/2601.18728) (arXiv 2026) [:octocat:](https://github.com/wdiepeveen/Riemannian-AmbientFlow)
- [Self-diffusion for Solving Inverse Problems](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7045440a0d224077ab1ee2766fa59af2-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/ggluo/self-diffusion)
- [DiffEM: Learning from Corrupted Data with Diffusion Models via Expectation Maximization](https://arxiv.org/abs/2510.12691) (arXiv 2025) [:octocat:](https://github.com/danialht/DiffEM)
- [Ambient Diffusion Omni: Training Good Models with Bad Data](https://proceedings.neurips.cc/paper_files/paper/2025/hash/369d566bd6419aeaf5a02c2be971391e-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/giannisdaras/ambient-omni)
- [Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models Trained on Corrupted Data](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2dd7f33ffbb59b4ff987be5442a13016-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/utcsilab/ambient-diffusion-mri)
- [Double Blind Imaging with Generative Modeling](https://arxiv.org/abs/2503.21501) (arXiv 2025)
- [Inverse Flow and Consistency Models](https://proceedings.mlr.press/v267/zhang25dm.html) (ICML 2025) [:octocat:](https://github.com/jzhoulab/InverseFlow)
- [Stochastic Forward-Backward Deconvolution: Training Diffusion Models with Finite Noisy Datasets](https://proceedings.mlr.press/v267/lu25n.html) (ICML 2025) [:octocat:](https://github.com/watml/SFBD)
- [Unsupervised Training of Convex Regularizers using Maximum Likelihood Estimation](https://openreview.net/forum?id=375tU7Tn0S) (TMLR 2024)
- [AmbientFlow: Invertible Generative Models from Incomplete, Noisy Measurements](https://openreview.net/forum?id=txpYITR8oa) (TMLR 2024) [:octocat:](https://github.com/comp-imaging-sci/AmbientFlow)
- [An Expectation-Maximization Algorithm for Training Clean Diffusion Models from Corrupted Observations](https://proceedings.neurips.cc/paper_files/paper/2024/hash/22912080c32a244c20bfc101033a6ac7-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/ai4imaging/EMDiffusoin)
- [GSURE-Based Diffusion Model Training with Corrupted Data](https://openreview.net/forum?id=BRl7fqMwaJ) (TMLR 2024) [:octocat:](https://github.com/bahjat-kawar/gsure-diffusion)
- [Learning Diffusion Priors from Observations by Expectation Maximization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9f94298bac4668db4dc77ddb0a244301-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/francois-rozet/diem)
- [Consistent Diffusion Meets Tweedie: Training Exact Ambient Diffusion Models with Noisy Data](https://proceedings.mlr.press/v235/daras24a.html) (ICML 2024) [:octocat:](https://github.com/giannisdaras/ambient-tweedie)

<a name="medical-imaging" />

### <u>Medical Imaging</u>

- [Establishing Stochastic Object Models from Noisy Data via Ambient Measurement-Integrated Diffusion](https://arxiv.org/abs/2512.14187) (arXiv 2025)
- [Measurement Score-Based MRI Reconstruction with Automatic Coil Sensitivity Estimation](https://arxiv.org/abs/2509.18402) (arXiv 2025)
- [Noise-Controllable Complex-Valued Diffusion Model for k-Space Data of Hyperpolarized 129Xe Lung MRI Generation](https://papers.miccai.org/miccai-2025/0640-Paper2171.html) (MICCAI 2025)
- [Stochastic Deep Restoration Priors for Imaging Inverse Problems](https://proceedings.mlr.press/v267/hu25s.html) (ICML 2025)
- [Robust Multi-Coil MRI Reconstruction via Self-Supervised Denoising](https://doi.org/10.1002/mrm.30591) (MRM 2025) [:octocat:](https://github.com/utcsilab/gsure-diffusion-mri)
- [Score-Based Diffusion Models With Self-Supervised Learning for Accelerated 3D Multi-Contrast Cardiac MR Imaging](https://doi.org/10.1109/TMI.2025.3534206) (TMI 2025)
- [A Self-supervised Diffusion Bridge for MRI Reconstruction](https://doi.org/10.1109/ISBI60581.2025.10980710) (ISBI 2025)

<a name="scientific-imaging" />

### <u>Scientific Imaging</u>

- [S3-PnP: Plug-and-Play Denoising for Scientific Imaging with Self-Supervised Score Priors](https://arxiv.org/abs/2610.18458) (arXiv 2026)
- [Hyperspectral Diffusion Equivariant Imaging (HyDiff-EI): A Self-supervised Framework for Hyperspectral Image Inpainting](https://arxiv.org/abs/2608.26812) (arXiv 2026)
- [Ambient Physics: Training Neural PDE Solvers with Partial Observations](https://icml.cc/virtual/2026/72595) (ICML 2026)
- [A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](https://neurips.cc/virtual/2025/poster/119352) (NeurIPS 2025) [:octocat:](https://github.com/swagnercarena/ddprism)
- [Self-supervised Axial Super-Resolution for Volume Microscopy via Diffusion-Guided Structure Distillation](https://papers.miccai.org/miccai-2025/0814-Paper2453.html) (MICCAI 2025) [:octocat:](https://github.com/hmzawz2/D2R-models)
- [CryoDRGN-AI: Neural Ab Initio Reconstruction of Challenging Cryo-EM and Cryo-ET Datasets](https://doi.org/10.1038/s41592-025-02720-4) (Nature Methods 2025) [:octocat:](https://github.com/ml-struct-bio/drgnai)
- [Self-Supervised Diffusion Model for 3-D Seismic Data Reconstruction](https://doi.org/10.1190/geo2024-0408.1) (Geophysics 2025)
- [Mixture of Neural Fields for Heterogeneous Reconstruction in Cryo-EM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6860d98b7ba3f331eaf03f5199c2b0be-Abstract.html) (NeurIPS 2024) [:octocat:](https://github.com/ml-struct-bio/hydra)
- [CryoSTAR: Leveraging Structural Priors and Constraints for Cryo-EM Heterogeneous Reconstruction](https://doi.org/10.1038/s41592-024-02486-1) (Nature Methods 2024) [:octocat:](https://github.com/bytedance/cryostar)
- [Reference-free Axial Super-resolution of 3D Microscopy Images using Implicit Neural Representation with a 2D Diffusion Prior](https://papers.miccai.org/miccai-2024/642-Paper2483.html) (MICCAI 2024) [:octocat:](https://github.com/hvcl/INR-diffusion)
- [denoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising](https://doi.org/10.1007/978-3-031-72691-0_13) (ECCV 2024) [:octocat:](https://github.com/juglab/denoiSplit)
- [CryoDRGN-ET: Deep Reconstructing Generative Networks for Visualizing Dynamic Biomolecules Inside Cells](https://doi.org/10.1038/s41592-024-02340-4) (Nature Methods 2024) [:octocat:](https://github.com/ml-struct-bio/cryodrgn)
- [Learning Structural Heterogeneity from Cryo-Electron Sub-Tomograms with tomoDRGN](https://doi.org/10.1038/s41592-024-02210-z) (Nature Methods 2024) [:octocat:](https://github.com/bpowell122/tomodrgn)

<a name="missing-data" />

### <u>Missing / Incomplete Data</u>

- [Generative Modeling under Non-Monotone MAR Missingness via Approximate Wasserstein Gradient Flows](https://openreview.net/forum?id=5SVmGGtTIM) (NeurIPS 2026)
- [AugMask: Training Diffusion Models on Incomplete Tabular Data via Stochastic Augmentation and Masking](https://proceedings.mlr.press/v306/kim26bh.html) (ICML 2026) [:octocat:](https://github.com/normal-kim/AugMask)
- [Incomplete Data, Complete Dynamics: A Diffusion Approach](https://proceedings.iclr.cc/paper_files/paper/2026/hash/123a18dfd821c8b440f42a00a27648d6-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/LOGO-CUHKSZ/incomplete-data-complete-dynamics)
- [MissScore: High-Order Score Estimation in the Presence of Missing Data](https://proceedings.mlr.press/v267/liu25aa.html) (ICML 2025)
- [Score Matching with Missing Data](https://proceedings.mlr.press/v267/givens25a.html) (ICML 2025) [:octocat:](https://github.com/joshgivens/ScoreMatchingwithMissingData)
- [DiffPuter: Empowering Diffusion Models for Missing Data Imputation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9e2ad37b8899366767df9319efc6f4de-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/hengruizhang98/DiffPuter)

<a name="contaminated-data" />

### <u>Contaminated, Outlier & Harmful Data</u>

- [Why Adversarially Train Diffusion Models?](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5a5ddf0ab751861025c00700093c5677-Abstract-Conference.html) (ICLR 2026)
- [Robust Score Matching](https://proceedings.mlr.press/v258/schwank25a.html) (AISTATS 2025) [:octocat:](https://gitlab.lrz.de/robust_score_matching/reproducibility)
- [A High-Quality Robust Diffusion Framework for Corrupted Dataset](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11188_ECCV_2024_paper.php) (ECCV 2024) [:octocat:](https://github.com/VinAIResearch/RDUOT)
- [Turning Waste into Wealth: Leveraging Low-Quality Samples for Enhancing Continuous Conditional Generative Adversarial Networks](https://ojs.aaai.org/index.php/AAAI/article/view/29065) (AAAI 2024) [:octocat:](https://github.com/UBCDingXin/Dual-NDA)

<a name="imbalanced-biased" />

### <u>Imbalanced, Long-Tailed & Biased Data</u>

- [Reweighted Flow Matching via Unbalanced Optimal Transport for Label-free Long-tailed Generation](https://openreview.net/forum?id=Z0OhxD9sBT) (NeurIPS 2026)
- [Stay Fair! Ensuring Group Fairness in Diffusion Models Across Guidance Scales](https://openreview.net/forum?id=tIDWhFyqX7) (NeurIPS 2026)
- [The Interplay of Data Structure and Imbalance in the Learning Dynamics of Diffusion Models](https://openreview.net/forum?id=3xmpzYAkAo) (NeurIPS 2026)
- [Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion](https://proceedings.mlr.press/v306/park26z.html) (ICML 2026) [:octocat:](https://github.com/soobin-um/jepa-guidance)
- [Equalized Generative Treatment: Matching f-divergences for Fairness in Generative Models](https://proceedings.mlr.press/v306/verine26a.html) (ICML 2026)
- [Improving Diffusion Models for Class-imbalanced Training Data via Capacity Manipulation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/75c411b0a06fa9e78f2a516b57b2ce62-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/Feng-Hong/ImbDiff-CM)
- [CORAL: Disentangling Latent Representations in Long-Tailed Diffusion](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a32539cb16274581a17e679f6046f4bf-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/SankarLab/coral-lt-diffusion)
- [PoGDiff: Product-of-Gaussians Diffusion Models for Imbalanced Text-to-Image Generation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ca03ccaf61b63cf3ee5de18140f5885b-Abstract-Conference.html) (NeurIPS 2025)
- [Principled Long-Tailed Generative Modeling via Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9ff5493d7d4a4b4002d67738237abd5b-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/pranoydas51/IGD-ML)
- [Imbalance in Balance: Online Concept Balancing in Generation Models](https://openaccess.thecvf.com/content/ICCV2025/html/Shi_Imbalance_in_Balance_Online_Concept_Balancing_in_Generation_Models_ICCV_2025_paper.html) (ICCV 2025) [:octocat:](https://github.com/KlingAIResearch/IMBA-Loss)
- [Boost-and-Skip: A Simple Guidance-Free Diffusion for Minority Generation](https://proceedings.mlr.press/v267/um25b.html) (ICML 2025) [:octocat:](https://github.com/soobin-um/BnS)
- [Minority-Focused Text-to-Image Generation via Prompt Optimization](https://openaccess.thecvf.com/content/CVPR2025/html/Um_Minority-Focused_Text-to-Image_Generation_via_Prompt_Optimization_CVPR_2025_paper.html) (CVPR 2025) [:octocat:](https://github.com/soobin-um/MinorityPrompt)
- [Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance](https://proceedings.iclr.cc/paper_files/paper/2025/hash/262ba7d20578417435414ce00e69fb12-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/krafton-ai/Rare-to-Frequent)
- [Diverse Rare Sample Generation with Pretrained GANs](https://ojs.aaai.org/index.php/AAAI/article/view/32480) (AAAI 2025)
- [Fair Text-to-Image Diffusion via Fair Mapping](https://ojs.aaai.org/index.php/AAAI/article/view/34823) (AAAI 2025)
- [DiffuLT: Diffusion for Long-tail Recognition Without External Knowledge](https://proceedings.neurips.cc/paper_files/paper/2024/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html) (NeurIPS 2024)
- [Rethinking Noise Sampling in Class-Imbalanced Diffusion Models](https://doi.org/10.1109/TIP.2024.3485484) (TIP 2024)
- [Self-Guided Generation of Minority Samples Using Diffusion Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8641_ECCV_2024_paper.php) (ECCV 2024) [:octocat:](https://github.com/soobin-um/sg-minority)
- [Balancing Act: Distribution-Guided Debiasing in Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/html/Parihar_Balancing_Act_Distribution-Guided_Debiasing_in_Diffusion_Models_CVPR_2024_paper.html) (CVPR 2024) [:octocat:](https://github.com/rishubhpar/debiasing_gen_models)
- [Taming the Tail in Class-Conditional GANs: Knowledge Sharing via Unconditional Training at Lower Resolutions](https://openaccess.thecvf.com/content/CVPR2024/html/Khorram_Taming_the_Tail_in_Class-Conditional_GANs_Knowledge_Sharing_via_Unconditional_CVPR_2024_paper.html) (CVPR 2024) [:octocat:](https://github.com/khorrams/utlo)
- [Don't Play Favorites: Minority Guidance for Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5dc387343572bb95f264e05b66e83951-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/soobin-um/minority-guidance)
- [Finetuning Text-to-Image Diffusion Models for Fairness](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6d0bf1265ea9635fb4f9d56f16d7efb2-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/sail-sg/finetune-fair-diffusion)
- [Long-tailed Diffusion Models with Oriented Calibration](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7dff60c3db3c0d2f3acf92f13b1b2472-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/MediaBrain-SJTU/OC_LT)
- [Training Unbiased Diffusion Models From Biased Dataset](https://proceedings.iclr.cc/paper_files/paper/2024/hash/89bd6217280d1417370c89ee493ba3c7-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/alsdudrla10/TIW-DSM)
- [Generating Images of Rare Concepts Using Pre-trained Diffusion Models](https://ojs.aaai.org/index.php/AAAI/article/view/28270) (AAAI 2024) [:octocat:](https://github.com/dvirsamuel/SeedSelect)

<a name="limited-data" />

### <u>Limited Data</u>

- [WILD-Diffusion: A WDRO Inspired Training Method for Diffusion Models under Limited Data](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9f709622912e87f8d84bffb097ae5503-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/USTC-DIAG-LAB/Wild-Diffusion)
- [SAR-DisentDM: A Semantic-Disentangled Diffusion Model for Limited-Data SAR Image Synthesis](https://ojs.aaai.org/index.php/AAAI/article/view/38165) (AAAI 2026)
- [Re-GAN: Data-Efficient GANs Training via Architectural Reconfiguration](https://doi.org/10.1109/TPAMI.2025.3590650) (TPAMI 2025) [:octocat:](https://github.com/IntellicentAI-Lab/Re-GAN)
- [Training Diffusion-based Generative Models with Limited Data](https://proceedings.mlr.press/v267/zhang25z.html) (ICML 2025) [:octocat:](https://github.com/zzhang05/LD-Diffusion)
- [Improving the Training of Data-Efficient GANs via Quality Aware Dynamic Discriminator Rejection Sampling](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Improving_the_Training_of_Data-Efficient_GANs_via_Quality_Aware_Dynamic_CVPR_2025_paper.html) (CVPR 2025) [:octocat:](https://github.com/zzhang05/QADDRS)
- [Style Quantization for Data-Efficient GAN Training](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Style_Quantization_for_Data-Efficient_GAN_Training_CVPR_2025_paper.html) (CVPR 2025)
- [Few-Shot Diffusion Models Escape the Curse of Dimensionality](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7eb6233e02f7d9efbb84acd839a996fb-Abstract-Conference.html) (NeurIPS 2024)
- [Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3195_ECCV_2024_paper.php) (ECCV 2024)
- [MS³D: A RG Flow-Based Regularization for GAN Training with Limited Data](https://proceedings.mlr.press/v235/wang24af.html) (ICML 2024)
- [CHAIN: Enhancing Generalization in Data-Efficient GANs via lipsCHitz continuity constrAIned Normalization](https://openaccess.thecvf.com/content/CVPR2024/html/Ni_CHAIN_Enhancing_Generalization_in_Data-Efficient_GANs_via_lipsCHitz_continuity_constrAIned_CVPR_2024_paper.html) (CVPR 2024) [:octocat:](https://github.com/MaxwellYaoNi/CHAIN)
- [High-Quality and Diverse Few-Shot Image Generation via Masked Discrimination](https://doi.org/10.1109/TIP.2024.3385295) (TIP 2024)
- [RG-GAN: Dynamic Regenerative Pruning for Data-Efficient Generative Adversarial Networks](https://ojs.aaai.org/index.php/AAAI/article/view/28271) (AAAI 2024) [:octocat:](https://github.com/IntellicentAI-Lab/RG-GAN)
- [WeditGAN: Few-Shot Image Generation via Latent Space Relocation](https://ojs.aaai.org/index.php/AAAI/article/view/27932) (AAAI 2024) [:octocat:](https://github.com/Ldhlwh/WeditGAN)

<a name="synthetic-data" />

### <u>Synthetic Data & Model Collapse</u>

- [Recursively Trained Diffusion Models: Limiting Collapse Distribution and Spectral Characterization](https://openreview.net/forum?id=TX42ZoaFHi) (NeurIPS 2026)
- [Self-Consuming Generative Models with Co-Evolving Human Preferences](https://openreview.net/forum?id=tsXMIKjzrO) (NeurIPS 2026) [:octocat:](https://github.com/osu-srml/Model-Reward-Coevolution)
- [Stability and Diversity of Networked Self-Consuming Generative Ecosystems](https://openreview.net/forum?id=Mq8Ixa9IkT) (NeurIPS 2026)
- [Curated Synthetic Data Doesn’t Have to Collapse: A Theoretical Study of Generative Retraining with Pluralistic Preferences](https://proceedings.mlr.press/v306/falahati26a.html) (ICML 2026)
- [Quantifying Error Propagation and Model Collapse in Diffusion Models](https://proceedings.mlr.press/v306/khelifa26a.html) (ICML 2026)
- [When Sample Selection Bias Precipitates Model Collapse](https://proceedings.mlr.press/v306/qiao26c.html) (ICML 2026) [:octocat:](https://github.com/XinbaoQiao/When-Sample-Selection-Bias-Precipitates-Model-Collapse)
- [When and How Human Curation Backfires: Preference Alignment under Multi-Model Self-Consuming Loop](https://proceedings.mlr.press/v306/zhang26gm.html) (ICML 2026) [:octocat:](https://github.com/osu-srml/curationBackfire)
- [Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence](https://proceedings.iclr.cc/paper_files/paper/2026/hash/29806b2e4dcb35e42abed21d43e2b7d8-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/liuqiyuanhhh/Verified-Synthetic-Data)
- [Neon: Negative Extrapolation From Self-Training Improves Image Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a9456f2962b306f99be4a0f99448df4b-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/VITA-Group/Neon)
- [What happens when generative AI models train recursively on each others' outputs?](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e5e38389c8c21a151ea898f877be9f6c-Abstract-Conference.html) (ICLR 2026)
- [Stabilizing Self-Consuming Diffusion Models with Latent Space Filtering](https://ojs.aaai.org/index.php/AAAI/article/view/39067) (AAAI 2026) [:octocat:](https://github.com/osu-srml/Latent-Space-Filtering)
- [A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](https://proceedings.neurips.cc/paper_files/paper/2025/hash/39d3b980155d3d52613e412b018004a4-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/shilianghe007/Model_Collapse)
- [Self-Verification Provably Prevents Model Collapse in Recursive Synthetic Training](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3380e8116452e0efbf36f35d95e88c94-Abstract-Conference.html) (NeurIPS 2025)
- [Collapse or Thrive: Perils and Promises of Synthetic Data in a Self-Generating World](https://proceedings.mlr.press/v267/kazdan25a.html) (ICML 2025) [:octocat:](https://github.com/RylanSchaeffer/KoyejoLab-Collapse-or-Thrive)
- [Self-Consuming Generative Models with Adversarially Curated Data](https://proceedings.mlr.press/v267/wei25o.html) (ICML 2025) [:octocat:](https://github.com/osu-srml/Adversarial-Curation)
- [Rate of Model Collapse in Recursive Training](https://proceedings.mlr.press/v258/suresh25a.html) (AISTATS 2025) [:octocat:](https://github.com/berserank/rate-of-model-collapse)
- [A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e92e6d20f7b5db5c2a1b23964d170fe2-Abstract-Conference.html) (ICLR 2025)
- [Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b9e88ae0308cf82d0b0f634ddbdf809a-Abstract-Conference.html) (NeurIPS 2024)
- [Self-Correcting Self-Consuming Loops for Generative Model Training](https://proceedings.mlr.press/v235/gillman24a.html) (ICML 2024) [:octocat:](https://github.com/nate-gillman/self-correcting-self-consuming)
- [Towards Theoretical Understandings of Self-Consuming Generative Models](https://proceedings.mlr.press/v235/fu24i.html) (ICML 2024)
- [On the Stability of Iterative Retraining of Generative Models on their own Data](https://proceedings.iclr.cc/paper_files/paper/2024/hash/339cab2a98edcba2d4ac42e5f8b64218-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/QB3/gen_models_dont_go_mad)
- [Self-Consuming Generative Models Go MAD](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ebc042e767de551803ccfcc45e2454f5-Abstract-Conference.html) (ICLR 2024)

<a name="imperfect-labels-conditions" />

## <u>Imperfect Labels & Conditions</u>

<a name="noisy-labels" />

### <u>Noisy Labels</u>

- [Robust Flow Matching under Target Corruption and Label Noise](https://openreview.net/forum?id=IjeT9T6KVY) (NeurIPS 2026)
- [Guiding Noisy Label Conditional Diffusion Models with Score-based Discriminator Correction](https://openaccess.thecvf.com/content/ICCV2025/html/Cong_Guiding_Noisy_Label_Conditional_Diffusion_Models_with_Score-based_Discriminator_Correction_ICCV_2025_paper.html) (ICCV 2025)
- [Robust Learning of Diffusion Models with Extremely Noisy Conditions](https://arxiv.org/abs/2510.10149) (arXiv 2025) [:octocat:](https://github.com/trustmlyoungscientist/Robust_Diffusion_Policy)
- [Label-Noise Robust Diffusion Models](https://openreview.net/forum?id=HXWTXXtHNl) (ICLR 2024) [:octocat:](https://github.com/byeonghu-na/tdsm)
- [Slight Corruption in Pre-training Data Makes Better Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e45c8d054739d31676619e7e11327f68-Abstract-Conference.html) (NeurIPS 2024)

<a name="noisy-captions" />

### <u>Noisy Captions & Text-Image Misalignment</u>

- [Is Your Text-to-Image Model Robust to Caption Noise?](https://openaccess.thecvf.com/content/CVPR2026F/html/Yu_Is_Your_Text-to-Image_Model_Robust_to_Caption_Noise_CVPRF_2026_paper.html) (CVPR Findings 2026)
- [SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](https://openaccess.thecvf.com/content/ICCV2025/html/Li_SuperEdit_Rectifying_and_Facilitating_Supervision_for_Instruction-Based_Image_Editing_ICCV_2025_paper.html) (ICCV 2025) [:octocat:](https://github.com/bytedance/SuperEdit)
- [What If We Recaption Billions of Web Images with LLaMA-3?](https://proceedings.mlr.press/v267/li25ch.html) (ICML 2025) [:octocat:](https://github.com/UCSC-VLAA/Recap-DataComp-1B)
- [Multi-Reward as Condition for Instruction-based Image Editing](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc82fe357440ce3f43e1939f4f795478-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/bytedance/Multi-Reward-Editing)
- [Revisit Large-Scale Image-Caption Data in Pre-training Multimodal Foundation Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ebba182cb97864368fdb6ae00773a5e4-Abstract-Conference.html) (ICLR 2025)
- [Don't Drop Your Samples! Coherence-Aware Training Benefits Conditional Diffusion](https://openaccess.thecvf.com/content/CVPR2024/html/Dufour_Dont_Drop_Your_Samples_Coherence-Aware_Training_Benefits_Conditional_Diffusion_CVPR_2024_paper.html) (CVPR 2024) [:octocat:](https://github.com/nicolas-dufour/CAD)

<a name="weak-supervision" />

### <u>Weak Supervision (Partial / Complementary / PU / Semi-Supervised)</u>

- [Asymptotically Exact Negative Guidance of Diffusion Models via Positive-Unlabeled Learning](https://openreview.net/forum?id=xCdsktvB3f) (NeurIPS 2026)
- [DELTA: Robustly Training Label-Conditional Diffusion Models with Weak Annotations](https://openreview.net/forum?id=1fJGfUpQRC) (NeurIPS 2026)
- [StableMaterials: Enhancing Diversity in Material Generation via Semi-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Vecchio_StableMaterials_Enhancing_Diversity_in_Material_Generation_via_Semi-Supervised_Learning_CVPR_2026_paper.html) (CVPR 2026)
- [Hybrid Layout Control for Diffusion Transformer: Fewer Annotations, Superior Aesthetics](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Hybrid_Layout_Control_for_Diffusion_Transformer_Fewer_Annotations_Superior_Aesthetics_ICCV_2025_paper.html) (ICCV 2025)
- [Positive-Unlabeled Diffusion Models for Preventing Sensitive Data Generation](https://openreview.net/forum?id=QQS7TudonJ) (ICLR 2025) [:octocat:](https://github.com/takahashihiroshi/pudm)

<a name="imperfect-conditions" />

### <u>Imperfect Conditions</u>

- [PathDiff: Histopathology Image Synthesis with Unpaired Text and Mask Conditions](https://openaccess.thecvf.com/content/ICCV2025/html/Bhosale_PathDiff_Histopathology_Image_Synthesis_with_Unpaired_Text_and_Mask_Conditions_ICCV_2025_paper.html) (ICCV 2025) [:octocat:](https://github.com/bhosalems/PathDiff)
- [Ctrl-U: Robust Conditional Image Generation via Uncertainty-aware Reward Modeling](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d76f8ea185a09f1dbb57a3365a4af768-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/grenoble-zhang/Ctrl-U)
- [When ControlNet Meets Inexplicit Masks: A Case Study of ControlNet on its Contour-following Ability](https://doi.org/10.1145/3664647.3680692) (ACM MM 2024) [:octocat:](https://github.com/DREAMXFAR/Shape-aware-ControlNet)
- [SmartControl: Enhancing ControlNet for Handling Rough Visual Conditions](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6412_ECCV_2024_paper.php) (ECCV 2024) [:octocat:](https://github.com/liuxiaoyu1104/SmartControl)
- [It's All About Your Sketch: Democratising Sketch Control in Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/html/Koley_Its_All_About_Your_Sketch_Democratising_Sketch_Control_in_Diffusion_CVPR_2024_paper.html) (CVPR 2024) [:octocat:](https://github.com/subhadeepkoley/StableSketching)

<a name="imperfect-feedback-rewards" />

## <u>Imperfect Feedback & Rewards</u>

<a name="noisy-preferences" />

### <u>Noisy Preferences</u>

- [Debiased DPO for Diffusion Models](https://openreview.net/forum?id=mVm3paaO7B) (NeurIPS 2026)
- [When Preference Labels Fall Short: Aligning Diffusion Models from Real Data](https://openreview.net/forum?id=KyAEhJSyEC) (ICML 2026) [:octocat:](https://github.com/Cwyxx/RealAlign)
- [CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_CRAFT_Aligning_Diffusion_Models_with_Fine-Tuning_Is_Easier_Than_You_CVPR_2026_paper.html) (CVPR 2026)
- [$\alpha$-DPO: Robust Preference Alignment for Diffusion Models via $\alpha$ Divergence](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a4c17d9b88eaefc9bdf7c656ffc8ce55-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/yangli-lab/Diffusion_alpha-DPO_ICLR2026)
- [Learning from Noisy Preferences: A Semi-Supervised Learning Approach to Direct Preference Optimization](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17061a94c3c7fda5fa24bbdd1832fa99-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/xinxinliu-ai/Semi-DPO)
- [ViPO: Visual Preference Optimization at Scale](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33bb58be3f0e903c75afa73d75b5c67e-Abstract-Conference.html) (ICLR 2026) [:octocat:](https://github.com/liming-ai/ViPO)
- [Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](https://proceedings.mlr.press/v267/lu25l.html) (ICML 2025) [:octocat:](https://github.com/JaydenLyh/SmPO)
- [Automated Filtering of Human Feedback Data for Aligning Text-to-Image Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e0c67da735023128cddefd8fc854282c-Abstract-Conference.html) (ICLR 2025)
- [When Preferences Diverge: Aligning Diffusion Models with Minority-Aware Adaptive DPO](https://arxiv.org/abs/2503.16921) (arXiv 2025)

<a name="imperfect-rewards" />

### <u>Imperfect Reward Models</u>

- [GARDO: Reinforcing Diffusion Models without Reward Hacking](https://openreview.net/forum?id=dmPzzDSb7m) (NeurIPS 2026) [:octocat:](https://github.com/tinnerhrhe/GARDO)
- [Manifold Drift in Flow Preference Optimization: A Root Cause of Reward Hacking](https://openreview.net/forum?id=9dO9vJ1V5d) (NeurIPS 2026)
- [Optimizing Visual Generative Models via Distribution-wise Rewards](https://openreview.net/forum?id=dxi8kGgKMM) (ICML 2026)
- [GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_GRPO-Guard_Mitigating_Implicit_Over-Optimization_in_Flow_Matching_via_Regulated_Clipping_CVPR_2026_paper.html) (CVPR 2026)
- [Reward Sharpness-Aware Fine-Tuning for Diffusion Models](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Reward_Sharpness-Aware_Fine-Tuning_for_Diffusion_Models_CVPR_2026_paper.html) (CVPR 2026)
- [SoliReward: Mitigating Susceptibility to Reward Hacking and Annotation Noise in Video Generation Reward Models](https://openaccess.thecvf.com/content/CVPR2026/html/Lian_SoliReward_Mitigating_Susceptibility_to_Reward_Hacking_and_Annotation_Noise_in_CVPR_2026_paper.html) (CVPR 2026) [:octocat:](https://github.com/lian700/SoliReward)
- [Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Taming_Preference_Mode_Collapse_via_Directional_Decoupling_Alignment_in_Diffusion_CVPR_2026_paper.html) (CVPR 2026)
- [The Image as Its Own Reward: Reinforcement Learning with Adversarial Reward for Image Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Mao_The_Image_as_Its_Own_Reward_Reinforcement_Learning_with_Adversarial_CVPR_2026_paper.html) (CVPR 2026) [:octocat:](https://github.com/showlab/Adv-GRPO)
- [Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](https://arxiv.org/abs/2603.12247) (arXiv 2026) [:octocat:](https://github.com/VisionXLab/FIRM-Reward)
- [Understanding Reward Hacking in Text-to-Image Reinforcement Learning](https://arxiv.org/abs/2601.03468) (arXiv 2026) [:octocat:](https://github.com/yq-hong/ArtifactReward)
- [Data-regularized Reinforcement Learning for Diffusion Models at Scale](https://arxiv.org/abs/2512.04332) (arXiv 2025)
- [Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning](https://arxiv.org/abs/2508.20751) (arXiv 2025) [:octocat:](https://github.com/CodeGoat24/Pref-GRPO)
- [Calibrated Multi-Preference Optimization for Aligning Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_Calibrated_Multi-Preference_Optimization_for_Aligning_Diffusion_Models_CVPR_2025_paper.html) (CVPR 2025)
- [Bridging Model-Based Optimization and Generative Modeling via Conservative Fine-Tuning of Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e68274fc4f158dbcbd4dddc672f7ee9c-Abstract-Conference.html) (NeurIPS 2024)
- [Confronting Reward Overoptimization for Diffusion Models: A Perspective of Inductive and Primacy Biases](https://proceedings.mlr.press/v235/zhang24ch.html) (ICML 2024) [:octocat:](https://github.com/ZiyiZhang27/tdpo)
- [Confidence-aware Reward Optimization for Fine-tuning Text-to-Image Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/275cabf15970186f9a4910702a77d357-Abstract-Conference.html) (ICLR 2024) [:octocat:](https://github.com/kykim0/TextNorm)

<a name="weak-feedback" />

### <u>Weak / Sparse Feedback</u>

- [Towards Fine-Grained Attribution: Instance-Aware Preference Optimization for Aligning Diffusion Models](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Towards_Fine-Grained_Attribution_Instance-Aware_Preference_Optimization_for_Aligning_Diffusion_Models_CVPR_2026_paper.html) (CVPR 2026)
- [Diffusion Negative Preference Optimization Made Simple](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4ee4857a7c7b6a45b75a6a30fdc38e3a-Abstract-Conference.html) (ICLR 2026)
- [SAIL: Self-Amplified Iterative Learning for Diffusion Model Alignment with Minimal Human Feedback](https://proceedings.iclr.cc/paper_files/paper/2026/hash/db5425cedbb9cc2b3b3b163b2326a65e-Abstract-Conference.html) (ICLR 2026)
- [Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](https://ojs.aaai.org/index.php/AAAI/article/view/37934) (AAAI 2026) [:octocat:](https://github.com/G-U-N/Diffusion-NPO)
- [Ranking-based Preference Optimization for Diffusion Models from Implicit User Feedback](https://proceedings.neurips.cc/paper_files/paper/2025/hash/561dd71d04695c85e1c0f6a0fbe0bd85-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/basiclab/DiffusionDRO)
- [Removing Concepts from Text-to-Image Models with Only Negative Samples](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e3cd4c3748274cfde024873d4ae91600-Abstract-Conference.html) (NeurIPS 2025)
- [Diffusion-NPO: Negative Preference Optimization for Better Preference Aligned Generation of Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b20852ac0a083262ecc98b49bca43086-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/G-U-N/Diffusion-NPO)
- [HERO: Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/88cc4c853ae933055f980ad9f9c02967-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/sony/hero)
- [Aligning Diffusion Models by Optimizing Human Utility](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2c487f8a54cf24c0684c32abc77fed56-Abstract-Conference.html) (NeurIPS 2024) [:octocat:](https://github.com/jacklishufan/diffusion-kto)
- [Feedback Efficient Online Fine-Tuning of Diffusion Models](https://proceedings.mlr.press/v235/uehara24a.html) (ICML 2024) [:octocat:](https://github.com/zhaoyl18/SEIKO)

<a name="theory-privacy-others" />

## <u>Theory, Privacy & Others</u>

<a name="theory" />

### <u>Theory & Analysis</u>

- [Robustness of Probabilistic Models to Low-Quality Data: A Multi-Perspective Analysis](https://proceedings.iclr.cc/paper_files/paper/2026/hash/445d4481c47fd47550240a5a941cc726-Abstract-Conference.html) (ICLR 2026)
- [How Much is a Noisy Image Worth? Data Scaling Laws for Ambient Diffusion](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45cc967d7899616f51993b7b363d35b5-Abstract-Conference.html) (ICLR 2025) [:octocat:](https://github.com/giannisdaras/ambient-laws)

<a name="memorization-privacy" />

### <u>Memorization, Privacy & Copyright</u>

- [Does Generation Require Memorization? Creative Diffusion Models using Ambient Diffusion](https://proceedings.mlr.press/v267/shah25b.html) (ICML 2025) [:octocat:](https://github.com/kulinshah98/memorization_noisy_data)

<a name="other" />

### <u>Others</u>

- [Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Beyond_the_Golden_Data_Resolving_the_Motion-Vision_Quality_Dilemma_via_CVPR_2026_paper.html) (CVPR 2026)
- [Ambient Proteins: Training Diffusion Models on Low Quality Structures](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5e9a29833416e5be3ef3a920489e3375-Abstract-Conference.html) (NeurIPS 2025) [:octocat:](https://github.com/jozhang97/ambient-proteins)
- [DIRECT-3D: Learning Direct Text-to-3D Generation on Massive Noisy 3D Data](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_DIRECT-3D_Learning_Direct_Text-to-3D_Generation_on_Massive_Noisy_3D_Data_CVPR_2024_paper.html) (CVPR 2024) [:octocat:](https://github.com/qihao067/direct3d)
<!-- papers:end -->
