# Model-card asset tools

Scripts that make the images (and the numbers behind them) on the LeafMachine3 Hugging Face model
cards. Canonical outputs live in `docs/hf_docs/<action>/` with a provenance JSON per set; the staged
model repos copy them into their own `docs/`. Re-run a script after retraining a model, then restage.

| Script | Card assets | Runs in |
|---|---|---|
| `leaf_segmenter.py` | `docs/hf_docs/leaf_segmenter/`: the input/output crop pairs (`io_*`) and the whole-specimen mask overlays (`overlay_*`) | LM3 runtime env |
| `landmark_detector.py` | `docs/hf_docs/landmark_detector/overlay_*`: Reporter Overlay_Landmarks renders for the pose model | LM3 runtime env |
| `landmark_detect_points.py` | `docs/hf_docs/landmark_detector/yolo26n_det_640/`: evaluation (box mAP + point metric sweep) and the predicted-point overlays + legend for the landmark-as-class detector | training env (ultralytics) |
| `plant_detector.py` | `docs/hf_docs/plant_detector/overlay_*`: the detector example grid, detected with the stage's own loader, settings and duplicate suppression, drawn with `build_summary_image` (boxes + labels) | LM3 runtime env |
| `birefnet_leaf.py` | `docs/hf_docs/leaf_segmenter/birefnet/panel_*`: the two BiRefNet leaf alternates (lamina + petiole, lamina only) and their combination on `Leaf_WHOLE` crops; the leaf_segmenter card's crops plus four more | LM3 runtime env (onnxruntime) |
| `specimen_detector.py` | `docs/hf_docs/specimen_detector/overlay_*`: Specimen boxes vs the hand-drawn ones on test sheets (not an LM3 stage, so its own drawing) | training env (ultralytics) |
| `specimen_segmenter.py` | `docs/hf_docs/specimen_segmenter/<key>/`: `examples` = the earlier UNet++ card's 20 sheets, Original \| UNet++ v1 (its saved masks) \| UNet++ v2; `failures` = sheets v1 failed on, both models through the stage's own path, raw masks and with paper removal (red = removed) | LM3 runtime env (onnxruntime) |
| `_common.py` | shared helpers: settings with the model path overridden, device pinning, read-only copies of the example runs' databases, "published only" selection | |

`leaf_segmenter.py` and `landmark_detector.py` reproduce the pipeline itself, so the images look like
LeafMachine3 output: the stage's own model loader and settings (`LM3_settings.yaml`, only the model path
overridden), the stage's coordinate re-basing, and the Reporter's renderers (`build_summary_image` with
masks only; `build_leaf_landmark_overlay` with the `landmark_measurements` panel). The specimens come from
the example runs named in each provenance file (`examples_out/...`), read from copies of their databases.
By default only the images already on the card are redrawn (files present in `docs/hf_docs/...` or in the
provenance `kept` list); `--all` also redraws pruned candidates so the set can be picked again.

```bash
# LM3 runtime env; ONNX on the GPU needs the venv's CUDA libs on LD_LIBRARY_PATH (else silent CPU fallback)
NV=$(ls -d .venv/lib/python3*/site-packages/nvidia); export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | tr '\n' ':')"
.venv/bin/python tools/release/model_card_assets/leaf_segmenter.py    --model <leaf seg model.onnx> --label "<model, iteration>" --device cuda:0
.venv/bin/python tools/release/model_card_assets/landmark_detector.py --model <pose model.onnx>     --label "<model, iteration>" --device cuda:0
```

Not here (they render from other runs or were made by hand):
- README images: `tools/release/build_readme_images.py`.
- The archival detector card grid (detector-only box overlays) and the ruler-classifier squarify
  before/after examples: their provenance JSONs in `docs/hf_docs/<action>/` record the sources.

Publishing: stage a repo folder (weights + exports hardlinked, card, `training_metadata.json`,
`manifest.json` with per-file sha256 and export-hash verification), then
`tools/release/hf_upload_verify.py <staged dir> phyloforfun/<repo> "<message>"`.
The iteration-2 staging scripts are in the training workspace (`it2_training/stage_it2_four.py`; the non-default
workflow models, BiRefNet leaf x2 and the Specimen detector: `it2_training/stage_workflow_models.py`).
