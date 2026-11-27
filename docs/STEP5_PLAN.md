# Step 5 — second baseline (DocTamper DTD) + second backbone (SegFormer-B2)

Goal: answer the two remaining Q1 objections: "only one published baseline" and "only ResNet-18".

## A. DocTamper detector (DTD) on the 613 GPT forged images

Environment: Kaggle notebook (Linux, GPU T4). Do NOT install into the local venv.

Notebook cells:

```python
# 1) code
!git clone https://github.com/triettu31251026766-del/ai-forged-doc.git
!pip install -q lmdb
```

```python
# 2) DocTamper model repo (official)
!git clone https://github.com/qcf-568/DocTamper.git
%cd DocTamper/models
!pip install -r requirements.txt   # if mmcv is pinned, install the Linux wheel; do not build from source
```

```python
# 3) run DTD over the image list
# The official repo expects its own test protocol; adapt the loader to read our list:
#   results/eval_lists/gpt_forged.txt  (613 stems)
# Save predicted masks as PNG in /kaggle/working/dtd_masks/<stem>.png at native size.
```

```python
# 4) metrics with our evaluator (same convention as TruFor)
!python ai-forged-doc/scripts/eval_trufor.py --npz-dir /kaggle/working/dtd_masks --list ai-forged-doc/results/eval_lists/gpt_forged.txt
```

Note: `eval_trufor.py` reads `.npz` with key `map`; if DTD outputs PNG masks, write a 20-line adapter
that loads the PNG, resizes to the ground-truth shape and thresholds at 0.5, then reuse the same IoU/F1 code.
Expected result: low IoU (0.05-0.20), since DTD targets splicing/copy-move text rather than generative inpainting.
That is the point of the comparison: it strengthens the "forensic detectors fail on AIGC" claim.

## B. SegFormer-B2 backbone

1. Extend `src/models/segmentation.py`:

```python
def build_segmentation(arch="resnet18", **kw):
    if arch == "segformer_b2":
        import segmentation_models_pytorch as smp
        return smp.Segformer(encoder_name="mit_b2", encoder_weights="imagenet",
                             in_channels=3, classes=1)
    ...existing resnet18 path...
```

2. Train the base model on Gemini (same recipe as `train_seg.py`, 384px, 10 epochs) with
   `--arch segformer_b2 --out results/seg_segformer`.
3. Fine-tune the patch segmenter with k=100 GPT labels (same command as `finetune_patch_seg.py`,
   with `--init results/seg_segformer/unet.pth`).
4. Evaluate with `eval_cascade_full.py --ckpt results/patch_seg_gpt_segformer/patch_seg.pth`.

Report both backbones in the same table. If SegFormer-B2 lands within +-0.03 IoU, the claim that
the phenomenon is not ResNet-specific is supported.

## C. Where the numbers go

- Table 6.9 (published baseline): add a DTD row.
- Table 6.8 / 6.10: add a "SegFormer-B2" row next to ResNet-18.
- Limitations: remove the "single architecture" and "DocTamper not runnable" sentences.

## D. Effort

| Task | Time |
|---|---|
| Kaggle env + DTD weights | 1-2 h |
| DTD inference 613 images | ~30 min GPU |
| Adapter + metrics | 1 h |
| SegFormer base train | ~2 h GPU |
| SegFormer patch fine-tune + cascade | ~1 h |
| Paper update | 2 h |
