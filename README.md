# Advances in Learning with Imperfect Information

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/> <img src="https://img.shields.io/badge/Number%20of%20Items-880-FF6F00" alt="PaperNum"/> ![Stars](https://img.shields.io/github/stars/wu-dd/Advances-in-Learning-with-Imperfect-Information?color=yellow&label=Stars) ![Forks](https://img.shields.io/github/forks/wu-dd/Advances-in-Learning-with-Imperfect-Information?color=green&label=Forks)

Curated lists of papers and code on **machine learning when the supervision is imperfect**: labels come as candidate sets or as labels an example does *not* have, training data are corrupted or incomplete, and the feedback used to align models is noisy or wrong.

### [🔍 Browse, search and filter all papers online](https://wu-dd.github.io/Advances-in-Learning-with-Imperfect-Information/)

**This project is curated and maintained by [Dong-Dong Wu](https://wu-dd.github.io/)**. It started in 2022 as *Advances in Partial/Complementary Label Learning* and now covers three topics. Suggestions are welcome, by email or through a pull request.

## Topics

<!-- stats:start -->
| Topic | What is imperfect | Papers | With code |
|---|---|---:|---:|
| [Partial & Complementary Label Learning](./partial-complementary-label-learning.md) | Classification when each training example comes with a set of candidate labels, only one of which is correct (partial labels), or with labels it does not belong to (complementary labels). | 331 | 174 |
| [Generative Modeling with Imperfect Information](./generative-modeling.md) | Generative models, mostly for images, trained from corrupted, incomplete, biased or weakly labeled data, or aligned with noisy feedback. | 175 | 106 |
| [LLM Post-Training with Imperfect Information](./llm-post-training.md) | Fine-tuning and aligning language models when preference labels, reward models, verifiers or demonstrations are noisy, weak, incomplete or corrupted. | 374 | 187 |
| **Total** | | **880** | **467** |
<!-- stats:end -->

Each list keeps papers from top conferences and journals, links the official paper page and, where one exists, the official code (:octocat:). Within each section the newest papers come first.

## Latest Updates
+ [2026/10/25] Renamed the project to *Advances in Learning with Imperfect Information*. Added two new lists, [Generative Modeling with Imperfect Information](./generative-modeling.md) and [LLM Post-Training with Imperfect Information](./llm-post-training.md), and an [interactive website](https://wu-dd.github.io/Advances-in-Learning-with-Imperfect-Information/).
+ [2026/10/24] Checked every entry of the partial/complementary label learning list: fixed links and venues, added official code links, reorganized the categories, and added 130 papers from 2024–2026.
+ [2026/03/16] Added 20+ missing partial/complementary label learning papers across NeurIPS/AAAI/TPAMI/PR/arXiv.
+ [2025/09/22] Updated papers published at conferences.
+ [2024/12/08] Updated code links of papers.
+ [2024/09/13] A major overhaul of the original GitHub repository.

## How to Contribute

All papers live in one file, [`data/papers.json`](./data/papers.json). To add a paper, append an entry with its `topic`, `section`, `title`, `url`, `venue`, `year` and `code` (or `null`), then run

```bash
python3 scripts/build.py
```

which regenerates the three Markdown lists and the counts above. The website reads the same file, so it updates automatically.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=wu-dd/Advances-in-Learning-with-Imperfect-Information&type=Date)](https://star-history.com/#wu-dd/Advances-in-Learning-with-Imperfect-Information&Date)

## Citing
If you find this project useful for your research, please use the following BibTeX entry.
```
@misc{Wu2022advances,
  author={Dong-Dong Wu},
  title={Advances in Learning with Imperfect Information},
  howpublished={\url{https://github.com/wu-dd/Advances-in-Learning-with-Imperfect-Information}},
  year={2022}
}
```
