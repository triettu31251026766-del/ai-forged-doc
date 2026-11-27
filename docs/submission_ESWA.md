# ESWA submission package

## Highlights (≤85 characters each)

1. A 100-label patch adaptation lifts AI-forged document IoU from 0.20 to 0.54.
2. Coarse peaks miss the field centre by 152 px; dense search recovers it.
3. A cascade needs 54 windows and 1.2 s per document without losing accuracy.
4. Conformal control certifies 9.8% image false negatives; triage FPR stays high.
5. TruFor reaches only 0.018 IoU on the same benchmark, versus 0.564 for TL-HRS.

## Cover letter

Dear Editor-in-Chief,

We submit our manuscript "Label-Efficient and Risk-Controlled Adaptation of
AI-Forged Document Detectors under Generator Shift" for consideration in
Expert Systems with Applications.

Financial documents are increasingly edited with generative inpainting, and
detectors trained on one generator collapse on the next. We show that with only
100 labelled samples a native-resolution patch segmenter raises localization
from 0.195 to 0.537 IoU on GPT-Image-2, and that the coarse stage used to place
that segmenter is the real bottleneck: it misses the field centre by 152 px
(median). Replacing it with a dense high-resolution cascade reaches 0.564 IoU
pooled with 54 windows and 1.04 s per document, and improves image-level triage
6.8x (Recall@1%FPR 0.042 to 0.286).

We further show that a published forensic detector (TruFor, CVPR 2023) obtains
only 0.018 IoU on exactly the same images. Finally, we apply conformal risk
control at document level: the image-level false-negative rate is certified at
9.8% for a 10% target, but the same operating point flags 56% of authentic
documents, so we state the guarantee as one-sided and do not claim that triage
is solved. Robustness to JPEG re-compression is quantified, and JPEG-augmented
adaptation raises Q90 IoU from 0.214 to 0.512.

The work fits ESWA's interest in applied intelligent systems with rigorous
evaluation. The manuscript is original, is not under review elsewhere, all
authors approved the submission, and code, splits and results are released.

Sincerely,
<author names and affiliations>
