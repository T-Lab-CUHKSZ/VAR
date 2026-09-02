<h2 align="center">
  <a href="https://arxiv.org/abs/2606.04483">Voices Across Registers: Corpus-Conditioned Vernacular Jailbreaks against Aligned LLMs via Fanfiction Subgenres</a>
</h2>

<h5 align="center">
  If our project helps you, please give us a star ⭐ and cite our <a href="#citation">paper</a>!
</h5>

<p align="center">
  <a href="assets/paper/VAR_EMNLP_2026.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b" alt="Paper PDF"></a>
  <a href="https://arxiv.org/abs/2606.04483"><img src="https://img.shields.io/badge/arXiv-2606.04483-b31b1b" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/EMNLP%202026-Main-blue" alt="EMNLP 2026 Main Conference">
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-green" alt="BibTeX citation"></a>
</p>

## News

- **[2026-08-21]** Our paper was accepted to the **EMNLP 2026 Main Conference**.
- **[2026-09-02]** Code and safely shareable verification artifacts are available in [`code/`](code/).

## Overview

Most jailbreak evaluations study discrete prompt templates whose surface forms can be fingerprinted and patched. We study a different axis: **register shift**. The same harmful behavior can be expressed through a natural writing register that an aligned model has learned well but its safety training may under-cover.

We instantiate this idea with twelve recurring writing registers drawn from Archive of Our Own (AO3) fanfiction. Our experiments cover **eight aligned LLMs**, **290 unique behaviors** from HarmBench and JailbreakBench, **twelve registers**, and a **four-judge ensemble**. We also introduce **VAR-A4**, a fixed four-turn extension that uses no adversarial attacker LLM and no per-target optimization during target interaction.

<p align="center">
  <img src="assets/readme/method_overview.png" width="96%" alt="VAR experimental framework">
</p>

### Research questions

1. Does a natural writing register increase attack success beyond existing jailbreak structures?
2. Can the effect be separated from prompt length and structural-template confounds?
3. Does register conditioning persist across models, hazards, defenses, and multi-turn dialogue?
4. Can a static four-turn progression outperform adaptive and longer alternatives?

## Method

### Single-turn vernacular attack

For each behavior-register pair, five passages sampled from the corresponding AO3 subgenre condition a target-agnostic creative-writing rewrite. The generated user prompt frames the evaluated behavior as the climax of a natural scene in that register.

The twelve register carriers are summarized below. They cover document-like, perspective-based, culturally specific, relational, and pacing-oriented narrative conventions.

<p align="center">
  <img src="assets/readme/register_taxonomy.png" width="82%" alt="The twelve AO3 registers">
</p>

### Factorial attribution

We cross the twelve registers with seven structural conditions, producing **84 cells**. Length-matched controls and behavior-by-model clustered GEE models separate the contribution of register from structure and prompt length.

<p align="center">
  <img src="assets/readme/factorial_results.png" width="76%" alt="Factorial experiment cell means">
</p>

### VAR-A4

VAR-A4 extends the strongest single-turn register into a fixed four-turn progression:

1. **Register entry:** establish the register-conditioned scene.
2. **Scene expansion:** continue the scene with concrete sensory and procedural detail.
3. **Specificity escalation:** increase specificity about actions, materials, and decisions.
4. **Compilation:** combine the preceding material into one chronological response.

The concatenated T1-T4 response is evaluated as a single output. The construction is static: it uses neither attacker-model feedback nor per-target optimization during target interaction.

<p align="center">
  <img src="assets/readme/vara4_pipeline.png" width="56%" alt="VAR-A4 four-turn pipeline">
</p>

## Main results

| Setting | Mean ASR | Summary |
|:--|--:|:--|
| Existing single-turn baselines | 0.278 | Six prior attack structures |
| Length-matched controls | 0.350 | Baselines rewritten to match vernacular prompt length |
| Vernacular attacks | **0.731** | Mean over twelve register arms |
| T1-only screenplay attack | 0.894 | Strongest single-turn register |
| VAR-A4 (screenplay) | **0.924** | Fixed four-turn sequence |

Across the eight models, vernacular prompts raise mean single-turn ASR from **0.278 to 0.731**, a **3.11x** vernacular-to-baseline ratio. The advantage remains after length matching. The factorial experiment further shows that register conditioning, rather than a particular jailbreak structure, drives the main gain.

<p align="center">
  <img src="assets/readme/single_turn_results.png" width="88%" alt="Per-arm single-turn attack success rates">
</p>

### Key result tables

<p align="center">
  <img src="assets/readme/table_01_single_turn_asr.png" width="96%" alt="Per-model and per-benchmark single-turn attack success rates">
</p>

<p align="center">
  <img src="assets/readme/table_02_factorial_attribution.png" width="48%" alt="Register meta-prompt comparison">
  <img src="assets/readme/table_06_multiturn_comparison.png" width="48%" alt="Multi-turn comparison">
</p>

VAR-A4 reaches **0.924 mean ASR**, exceeding Crescendo, FITD, ActorAttack, and the corresponding T1-only attack. Structural ablations show that compilation at T4 provides the largest marginal contribution among the tested turns; duplicating a turn with an added paraphrase does not improve the full sequence.

<p align="center">
  <img src="assets/readme/table_07_vara4_ablations.png" width="96%" alt="VAR-A4 structural ablations">
</p>

## Evaluation protocol

Every prompt-response pair receives four independent labels:

- HarmBench-13B binary classifier
- LlamaGuard-3-8B binary safety classifier
- WildGuard-7B binary safety classifier
- GPT-5.4-mini scoring the StrongREJECT rubric, thresholded at 0.25

The reported attack-success label uses a **two-of-four ensemble**. The paper additionally reports raw agreement, Cohen's kappa, PABAK, Gwet's AC1, a blinded 200-item human audit, cluster-bootstrap intervals, and StrongREJECT-threshold robustness.

## Code

The public implementation is available in [`code/`](code/). It includes corpus construction, attack generation, target inference, four-judge evaluation, defenses, baseline adapters, human-audit tooling, and analysis scripts. See [`code/README.md`](code/README.md) for setup and reproduction notes.

## Paper and figures

The camera-ready paper is available as [`VAR_EMNLP_2026.pdf`](assets/paper/VAR_EMNLP_2026.pdf). Publication-quality figure PDFs are provided in paper order:

<details>
<summary><strong>Show all figure files</strong></summary>

| Figure | Description | PDF |
|:--:|:--|:--:|
| 1 | Scenario comparison | [PDF](assets/paper/figures/Figure_01.pdf) |
| 2 | Experimental design framework | [PDF](assets/paper/figures/Figure_02.pdf) |
| 3 | VAR-A4 pipeline | [PDF](assets/paper/figures/Figure_03.pdf) |
| 4 | Per-arm single-turn ASR | [PDF](assets/paper/figures/Figure_04.pdf) |
| 5 | Factorial experiment cell means | [PDF](assets/paper/figures/Figure_05.pdf) |
| 6 | ASR by prompt-length bucket | [PDF](assets/paper/figures/Figure_06.pdf) |
| 7 | Four-judge Cohen's kappa matrices | [PDF](assets/paper/figures/Figure_07.pdf) |
| 8 | Five-shot register meta-prompt | [PDF](assets/paper/figures/Figure_08.pdf) |
| 9 | Twelve AO3 register carriers | [PDF](assets/paper/figures/Figure_09.pdf) |
| 10 | Worked VAR-A4 example | [PDF](assets/paper/figures/Figure_10.pdf) |
| 11 | Per-judge and ensemble ASR | [PDF](assets/paper/figures/Figure_11.pdf) |
| 12 | Single-turn examples, set 1 | [PDF](assets/paper/figures/Figure_12.pdf) |
| 13 | Single-turn examples, set 2 | [PDF](assets/paper/figures/Figure_13.pdf) |
| 14 | Multi-turn transcript, example 1 | [PDF](assets/paper/figures/Figure_14.pdf) |
| 15 | Multi-turn transcript, example 2 | [PDF](assets/paper/figures/Figure_15.pdf) |

</details>

## Responsible release

This repository supports safety research on aligned language models. Worked examples in the paper redact operationally sensitive spans. Source-derived exemplars and unredacted attack material that could enable direct misuse are not included in this release. The [`code/`](code/) directory provides the implementation and safely shareable aggregate artifacts for method inspection and verification.

## Citation

If this project is useful for your work, please cite:

```bibtex
@inproceedings{luo2026voices,
  title     = {Voices Across Registers: Corpus-Conditioned Vernacular Jailbreaks against Aligned LLMs via Fanfiction Subgenres},
  author    = {Luo, Zhongze and Shi, Ruihe and Yin, Zhenshuai and Liu, Haoyue and Wan, Weixuan and Tang, Xiaoying},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```


An earlier version is available as:

```bibtex
@article{luo2026off,
  title   = {Off-Distribution Voices: Fanfiction Subgenres as Universal Vernacular Jailbreaks for Aligned LLMs},
  author  = {Luo, Zhongze and Shi, Ruihe and Yin, Zhenshuai and Liu, Haoyue and Wan, Weixuan and Tang, Xiaoying},
  journal = {arXiv preprint arXiv:2606.04483},
  year    = {2026}
}
```
