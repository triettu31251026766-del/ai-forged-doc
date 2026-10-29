# Label-Efficient Adaptation of AI-Forged Document Detectors under Generator Shift via Rehearsal

**Authors:** 

---

## Abstract

Generative image editing has made it possible to alter a single numeric field in a financial document in under a second, and detectors trained on traditional (e.g., Photoshop-based) forgeries are largely blind to such edits. Even a detector trained on one generative inpainter collapses when the underlying generator changes — a *generator shift*. In realistic banking deployments, collecting many labelled forgeries for every new generator is infeasible. We study **label-efficient adaptation** of an AI-forged document detector under generator shift, and we ask whether a small number of labelled target-generator samples is sufficient to recover localization performance without catastrophically forgetting previously seen generators. Using the AIForge-Doc benchmark (Gemini, Ideogram, GPT-Image-2), we show that: (i) a detector trained on Gemini collapses on unseen generators (IoU 0.052 on GPT-Image-2); (ii) few-shot fine-tuning recovers localization monotonically, reaching IoU 0.195 ± 0.037 with only 100 labelled target samples (≈3.7× over zero-shot); (iii) training from scratch with the same 100 labels is substantially worse (IoU 0.085), showing that base-generator pretraining matters; and (iv) rehearsal of 100 old-generator samples restores in-domain performance to 0.672 ± 0.006 (near the 0.669 zero-shot level) at a modest target-generator cost (IoU 0.162). We further report a set of negative results: temperature scaling, global and per-image threshold calibration, sample-selection strategies, encoder freezing, frequency channels, and generic anomaly detection do **not** provide consistent gains. Our analysis indicates that the bottleneck is **localization precision**, not decision-threshold calibration. We release code, splits, and all results for reproducibility.

---

## 1. Introduction

Digital document fraud is increasing rapidly. Recent generative inpainting APIs can replace a number or a date in a photographed receipt while preserving font, colour, and texture, leaving no obvious compression or copy-move traces. Detectors designed for traditional document tampering therefore degrade substantially on such forgeries.

A second, less discussed problem is **generator shift**. A detector trained on forgeries produced by one inpainting model learns generator-specific artefacts rather than a generator-agnostic notion of tampering. When a new generator appears, the detector collapses. In a bank, this is operationally critical: the institution cannot wait to collect thousands of labelled forgeries for every new tool.

We therefore study a practical question:

> *Given a detector trained on an old generator, how many labelled samples from a new generator are needed to recover localization, and how can we adapt without forgetting the old generator?*

We use the AIForge-Doc benchmark, which provides AI-inpainted financial and form documents with pixel-precise masks for three generator families: Gemini 2.5 Flash Image, Ideogram v2 Edit, and GPT-Image-2. We adopt a leave-one-generator-out protocol and evaluate both localization (IoU/F1/Dice) and detection-oriented metrics (image AUC/AP, Recall@1%FPR).

**Contributions.**
1. A controlled study of label-efficient adaptation for AI-forged document detection under generator shift.
2. A few-shot adaptation curve (k = 0, 5, 25, 100; 3 seeds) showing monotonic recovery (IoU 0.052 → 0.195 ± 0.037).
3. A rehearsal strategy that restores old-generator performance to near zero-shot level (Gemini IoU 0.672 ± 0.006) while retaining most of the adaptation gain.
4. An honest set of negative results: calibration, adaptive thresholding, sample selection, encoder freezing, frequency channels, and generic anomaly detection do not provide consistent gains; we analyze why.
5. Qualitative evidence (score maps) showing that adaptation sharpens localization rather than merely shifting thresholds.

---

## 2. Related Work

**AI-forged document detection.** DocTamper introduced a large document-tampering benchmark and a document-specific detector based on frequency-domain losses. TruFor combines a transformer backbone with a noise fingerprint for general image forgery localization. AIForge-Doc and DOCFORGE-BENCH recently showed that these detectors degrade severely on diffusion-inpainted documents, and that the problem remains unsolved. Our work builds directly on AIForge-Doc and studies adaptation rather than zero-shot detection.

**Few-shot and domain adaptation.** Few-shot learning and unsupervised domain adaptation are mature fields. We do not propose a new adaptation algorithm; instead, we characterize how standard fine-tuning behaves in the specific regime of extreme label scarcity and generator shift, and identify which components (pretraining, rehearsal, calibration) actually matter.

**Continual learning and rehearsal.** Rehearsal (experience replay) is a classical remedy for catastrophic forgetting. We show that a small replay buffer of old-generator samples is sufficient to preserve in-domain performance while adapting to a new generator.

**Calibration.** Temperature scaling is a standard post-hoc calibration method. We find that although it changes confidence, it does not change ranking and therefore does not improve IoU; global and per-image threshold calibration are generator-specific and do not transfer reliably.

---

## 3. Dataset and Protocol

### 3.1 Dataset

We use **AIForge-Doc** (v1 and v2):

| Split | Forged | Authentic | Generators |
|---|---|---|---|
| v1 train | 3,249 | 3,249 | Gemini (2,912), Ideogram (337) |
| v1 test | 812 | 812 | Gemini (727), Ideogram (85) |
| v2 train | 2,453 | — | GPT-Image-2 |
| v2 test | 613 | — | GPT-Image-2 |

Documents are receipts and forms; forgeries replace a single numeric field. The median tampered area is ≈1% of the image. Masks are binary (0/255). We verified the integrity of all 18,315 files (0 corrupt) and visually confirmed mask alignment.

### 3.2 Leave-one-generator-out protocol

We define three settings:

- **P1 (zero-shot shift):** train on Gemini, test on Gemini (in-domain), Ideogram, and GPT-Image-2.
- **P2:** train on Gemini + Ideogram, test on GPT-Image-2 (unseen).
- **P3:** train on Gemini + GPT-Image-2, test on Ideogram (unseen).

Unless stated otherwise, the base detector is trained on Gemini (2,912 forged + 2,912 authentic).

### 3.3 Metrics

- **Localization:** IoU, F1, Dice at threshold 0.5.
- **Ranking:** pixel AUC (mean per image) and image AUC.
- **Detection under alert budget:** image Average Precision (AP) and Recall at FPR = 1% (Recall@1%FPR), which reflects a bank's limited review capacity.

---

## 4. Method

### 4.1 Base detector

We use a U-Net with a ResNet-18 encoder pretrained on ImageNet. The model outputs a single-channel logit map. It is trained on Gemini with:

\[
\mathcal{L} = \mathrm{BCEWithLogits}(\hat{y}, y; w=20) + \mathcal{L}_{\mathrm{Dice}},
\]

where the positive weight compensates for the extreme pixel imbalance (≈1% tampered). Inputs are letterboxed to 384×384 to preserve document aspect ratio. Training uses AdamW (lr \(10^{-4}\)) for 10 epochs.

### 4.2 Few-shot adaptation

Given \(k\) labelled samples from the target generator (plus \(k\) authentic samples), we fine-tune the base detector for 10 epochs with the same loss and optimizer. All models are evaluated on a disjoint target test split.

### 4.3 Rehearsal

To mitigate forgetting, we add \(r\) samples from the old generator (Gemini) to the fine-tuning set. We compare random replay with two informed selectors: **hard** (highest per-sample BCE under the base model) and **uncertainty** (highest predictive entropy).

### 4.4 Calibration (studied as an ablation)

We study three calibration strategies: temperature scaling fitted on a held-out fraction of the labelled target samples; global threshold optimization; and per-image adaptive thresholding (keep the top-\(p\)% pixels). These are evaluated but, as shown in Section 6.5, do not provide consistent gains.

---

## 5. Experimental Setup

- **Base training:** Gemini train split, 384×384, 10 epochs.
- **Adaptation:** \(k \in \{5, 25, 100\}\) forged + \(k\) authentic target samples, 10 epochs.
- **Rehearsal:** \(r = 100\) old-generator samples.
- **Seeds:** 3 seeds for the adaptation curve, 4 for rehearsal.
- **Hardware:** a single NVIDIA RTX 5060 Laptop GPU (8 GB).
- **Reproducibility:** fixed splits, seeds, and scripts are released.

---

## 6. Results

### 6.1 Zero-shot detector collapses under generator shift

| Test set | IoU | F1 | Dice | Image AUC |
|---|---|---|---|---|
| Gemini (in-domain) | 0.669 | 0.802 | 0.719 | 0.946 |
| Ideogram (unseen) | 0.098 | 0.179 | 0.109 | 0.546 |
| GPT-Image-2 (unseen) | 0.052 | 0.099 | 0.104 | 0.520 |

A detector trained on Gemini achieves strong in-domain localization but collapses on unseen generators. Notably, on GPT-Image-2 the *image-level* AUC is near chance, yet the **pixel AUC remains 0.899** (Section 6.5). In other words, the model still ranks tampered pixels above background, but its predicted mask is spatially imprecise.

### 6.2 Few-shot adaptation recovers localization

| k | IoU | F1 | Dice | Pixel AUC | Gemini IoU (forgetting) |
|---|---|---|---|---|---|
| 0 | 0.052 | 0.099 | 0.104 | 0.899 | 0.669 |
| 5 | 0.068 ± 0.015 | — | — | 0.877 ± 0.020 | 0.624 ± 0.017 |
| 25 | 0.105 ± 0.031 | — | — | 0.830 ± 0.023 | 0.577 ± 0.031 |
| 100 | **0.195 ± 0.037** | **0.292** | **0.202** | **0.907 ± 0.031** | **0.494 ± 0.054** |

Adaptation improves localization monotonically, reaching a **3.7×** IoU gain with only 100 labelled target samples. The cost is forgetting: in-domain Gemini drops from 0.669 to 0.494.

### 6.3 Pretraining matters: fine-tuning vs. training from scratch

| Setting (k=100) | GPT IoU | GPT F1 | Pixel AUC | Gemini IoU |
|---|---|---|---|---|
| Fine-tune from Gemini base | **0.1597** | **0.2754** | **0.8536** | **0.5522** |
| Train from scratch | 0.0850 | 0.1567 | 0.6977 | 0.0750 |

Fine-tuning is ≈1.9× better than training from scratch on the target generator, and preserves old-generator knowledge far better. Label-efficient adaptation therefore benefits from base-generator pretraining.

### 6.4 Rehearsal eliminates forgetting

| Method (k=100) | GPT IoU | Gemini IoU | Ideogram IoU |
|---|---|---|---|
| Standard fine-tune (3 seeds) | 0.195 ± 0.030 | 0.494 ± 0.044 | 0.096 ± 0.014 |
| **Replay 100 (4 seeds)** | **0.162 ± 0.021** | **0.672 ± 0.006** | 0.108 ± 0.015 |

Replay restores Gemini to **0.672 ± 0.006**, essentially the zero-shot level (0.669), at a modest target-generator cost (0.195 → 0.162). This is the central practical finding: a small rehearsal buffer is sufficient to prevent catastrophic forgetting under generator shift.

Random replay performed best among the selectors; hard and uncertainty-based replay did not improve retention at the same budget.

### 6.5 Negative results and ablations

| Strategy | Target (GPT) IoU | Old (Gemini) IoU | Conclusion |
|---|---|---|---|
| Global threshold tuning | 0.160 → 0.194 (one model) | — | modest, generator-specific |
| Temperature scaling | unchanged | unchanged | changes confidence, not ranking |
| Per-image adaptive threshold | 0.158 → 0.141 (p1%) | 0.555 → 0.365 | worse than fixed 0.5 |
| Sample selection (k=50) | random 0.157 vs k-center 0.134 vs uncertainty 0.126 | — | random is a strong baseline |
| Freeze encoder | 0.113 | 0.087 | fails |
| Augmentation (JPEG/blur/noise) | 0.052 → 0.054 | 0.669 → 0.589 | marginal, harms in-domain |
| Multi-generator training | P2 GPT 0.058; P3 Ideogram 0.121 | — | marginal |
| Frequency channels (FR-Net) | 0.061 | 0.561 | marginal, harms in-domain |
| PatchCore (ResNet-18 / DINOv2) | ≈0.05 (image AUC ≈0.5) | — | fails |

**Key observation.** Pixel AUC is consistently high (0.83–0.91) even when IoU is low, and neither temperature scaling nor per-image adaptive thresholding improves IoU. The bottleneck is therefore **spatial localization precision**, not decision-threshold calibration. This also explains why calibration tuned on the target generator can *hurt* the old generator: it is generator-specific.

### 6.6 Qualitative analysis

Figure `paper/figures/fig_xai_zero_shot_gpt.png` shows zero-shot score maps on GPT-Image-2: predictions are diffuse, often off-target, and sometimes nearly blank. After adaptation (`fig_xai_adapted_gpt.png`), score maps concentrate on the edited field. In-domain maps (`fig_xai_in_domain.png`) are sharp and closely match the ground truth. The qualitative evolution mirrors the quantitative gains.

---

## 7. Discussion

**Why does adaptation help?** Fine-tuning on a few target samples appears to sharpen the spatial response around the edited region, rather than merely shifting a global threshold. This is consistent with the high pixel AUC before adaptation and with the failure of threshold-calibration methods.

**Why does forgetting happen?** The base detector's features encode generator-specific artefacts. Fine-tuning on a new generator overwrites part of this encoding. Rehearsal of a small number of old samples anchors the old decision boundary and restores in-domain performance.

**Why do calibration and selection fail?** Calibration is generator-specific and does not change pixel ranking; sample selection does not address the underlying localization error. These negative results are useful: they rule out several intuitive but ineffective remedies.

---

## 8. Limitations

- **Small, narrow benchmark.** AIForge-Doc contains ≈4k forged receipts/forms; generalization to other document types, languages, and bank-specific data is unverified.
- **Three generators.** Gemini, Ideogram, and GPT-Image-2 do not cover the full space of generative editors.
- **Synthetic edits.** Forgeries replace a single numeric field; whole-document synthesis and signature forgery are out of scope.
- **Single architecture.** We use a ResNet-18 U-Net; larger or text-aware models may behave differently.
- **Modest absolute performance.** Even after adaptation, IoU remains far below in-domain levels, indicating that label-efficient adaptation is necessary but not sufficient.
- **Variance.** Results vary across seeds; we report mean ± std where possible.

---

## 9. Future Work

A natural extension is **federated adaptation**: when one bank encounters a new generator and collects a small labelled set, it could share the *model update* (not raw customer data) with other institutions, allowing them to defend against the same threat without violating data-protection regulations. Combining few-shot adaptation with secure aggregation and differential privacy is an important direction.

Other directions include: (i) text-aware and higher-resolution localization; (ii) continual adaptation over a stream of generators; (iii) stronger domain-generalization objectives; (iv) broader benchmarks across document types and languages; and (v) calibrated uncertainty for human-in-the-loop review.

---

## 10. Conclusion

We studied label-efficient adaptation of AI-forged document detectors under generator shift. A detector trained on one generator collapses on unseen generators, but a small number of labelled target samples recovers localization substantially, and a small rehearsal buffer preserves old-generator performance with negligible forgetting. Our analysis shows that the remaining bottleneck is localization precision rather than threshold calibration, and that several intuitive remedies (calibration, adaptive thresholding, sample selection, encoder freezing) do not provide consistent gains. We hope these results provide a realistic foundation for deploying and adapting document-fraud detectors in banking workflows.

---

## References

1. J. Wu, Y. Zhou, M. Xu, et al. *AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents.* arXiv:2602.20569, 2026.
2. Z. Zhao, W. Xia, E. Wei, et al. *DOCFORGE-BENCH: A Comprehensive 0-shot Benchmark for Document Forgery Detection and Analysis.* arXiv:2603.01433, 2026.
3. J. Wu, Y. Zhou, et al. *When the Forger Is the Judge: GPT-Image-2 Cannot Recognize Its Own Faked Documents.* arXiv:2604.25213, 2026.
4. C. Qu, C. Li, et al. *Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution.* CVPR, 2023. (DocTamper)
5. F. Guillaro, D. Cozzolino, et al. *TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization.* CVPR, 2023.
6. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
7. O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.
8. M. Oquab, T. Darcet, et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR, 2024.
9. K. Roth, L. Pemula, et al. *Towards Total Recall in Industrial Anomaly Detection.* CVPR, 2022. (PatchCore)
10. C. Guo, G. Pleiss, Y. Sun, K. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017.
11. M. P. Naeini, G. Cooper, M. Hauskrecht. *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI, 2015.
12. S.-A. Rebuffi, A. Kolesnikov, G. Sperl, C. Lampert. *iCaRL: Incremental Classifier and Representation Learning.* CVPR, 2017.
13. D. Rolnick, A. Ahuja, et al. *Experience Replay for Continual Learning.* NeurIPS, 2019.
14. B. McMahan, E. Moore, et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS, 2017. (FedAvg)
15. C. Finn, P. Abbeel, S. Levine. *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML, 2017.
16. J. Snell, K. Swersky, R. Zemel. *Prototypical Networks for Few-Shot Learning.* NeurIPS, 2017.
