<h2 align="center"> <a href="https://openreview.net/forum?id=UTxO0tkHU2">Off-Distribution Voices: Fanfiction Subgenres as Universal Vernacular Jailbreaks for Aligned LLMs</a></h2>
<h5 align="center"> If our project helps you, please give us a star ⭐ and cite our <a href="#citation">paper</a>!</h2>
<h5 align="center">

[![github](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/luozhongze/ODV)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Main%20Conference-blue)](https://arxiv.org/abs/2606.04483)
[![arXiv](https://img.shields.io/badge/arXiv-2606.04483-blue)](https://arxiv.org/abs/2606.04483)

## News
- **Code and Datasets are coming soon.**
- **[2026.08.21]**: 🎉 Our [paper](https://arxiv.org/abs/2606.04483) is accepted to the EMNLP 2026 Main Conference!

## Why Do We Need Off-Distribution Voices?

Most jailbreaks are discrete prompt artifacts whose surface forms can be fingerprinted and patched. We instead study **register shift**: harmful requests expressed through natural writing styles that safety alignment may under-cover.

We introduce a corpus-conditioned attack family based on twelve Archive of Our Own fanfiction subgenres. Across eight aligned LLMs and 290 behaviors from HarmBench and JailbreakBench, vernacular prompts raise mean attack success rate from **0.278 to 0.731** under a four-judge ensemble. A factorial design separates the effect of register from structure and length.

We further introduce **SAGA-A4**, a static four-turn pipeline requiring no adversarial attacker LLM or per-target optimization. It reaches **0.924 ASR** on its primary screenplay setting and improves over single-turn attacks across all twelve registers.

## Main Results

| Setting | Mean ASR |
| :----: | :--: |
| Single-turn baselines | 0.278 |
| Vernacular attacks | 0.731 |
| SAGA-A4 (screenplay) | 0.924 |

The rebuttal-stage analyses further show register classification accuracy up to **0.99**, ensemble precision of **1.00** in a 200-item human audit, stable results under cluster-bootstrap confidence intervals and a StrongREJECT threshold sweep, and multi-turn gains for every register.

## Method

### Single-turn Vernacular Attack

Five passages sampled from one fanfiction subgenre condition a target-agnostic creative-writing rewrite. The resulting prompt embeds the evaluated behavior as the climax of a natural scene.

### Factorial Attribution

An 84-cell design crosses register and prompt structure, with length-matched controls and GEE analysis, to isolate the contribution of register.

### SAGA-A4

SAGA-A4 extends the attack into a fixed four-turn narrative progression without adversarial attacker-model feedback or per-target optimization.

## Citation

If you find this project helpful, please consider citing our work:

```
@inproceedings{luo2026offdistribution,
  title={Off-Distribution Voices: Fanfiction Subgenres as Universal Vernacular Jailbreaks for Aligned LLMs},
  author={Luo, Zhongze and Shi, Ruihe and Yin, Zhenshuai and Liu, Haoyue and Wan, Weixuan and Tang, Xiaoying},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```
