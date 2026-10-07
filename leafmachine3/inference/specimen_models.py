"""The specimen-segmenter model registry: which model a settings file names, and how to feed it.

A specimen segmenter is chosen by NAME, never by inspecting its file. ``modules.specimen_segmenter
.model.key`` carries the name from the settings file to :func:`resolve_specimen_model`, the factory
hands the resolved :class:`SpecimenModelSpec` to the backend, and the backend runs that spec's input
workflow. The key is the model's Hugging Face repo suffix (``lm3_specimen_segmenter__<key>``), so a
retrained version, published into the same repo, keeps its key and its workflow.

Each workflow reproduces the image preparation its model was TRAINED with:

``letterbox_imagenet`` (UNet++)
    BGR -> RGB; keep aspect, long side -> ``imgsz`` with ``cv2.INTER_AREA``; centered on a white
    (255) square; /255 then ImageNet mean/std. Output: sigmoid, threshold at the square, drop the
    pad, nearest-neighbor resize to the image. Reference: the UNet++ trainer's
    ``common/seg_data.py: letterbox_pair`` and ``common/seg_infer.py: predict_mask``.
``stretch_imagenet`` (BiRefNet)
    Resize both sides to ``imgsz`` (aspect NOT kept) with ``cv2.INTER_LINEAR`` on the BGR image,
    then BGR -> RGB; /255 then ImageNet mean/std. Output: sigmoid, bilinear resize of the
    probabilities to the image, then threshold. Reference: BiRefNet's training loader
    ``utils.path_to_image`` (``cv2.resize(..., INTER_LINEAR)``) + ``ToTensor`` + ``Normalize``.
``ultralytics_seg`` (YOLO26-seg)
    Keep aspect, long side -> ``imgsz`` with ``cv2.INTER_LINEAR``; centered, padded with 114 to the
    next multiple of the stride; BGR -> RGB; /255. Output: end2end detections at ``yolo.conf``,
    instance masks at full image resolution, union = foreground. Reference: Ultralytics'
    ``LetterBox`` / ``BasePredictor.preprocess``, reproduced by ``inference/ultra_replacements.py``.

Adding a model = one registry entry (and a workflow, if it was trained differently from these).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

WORKFLOW_LETTERBOX_IMAGENET = "letterbox_imagenet"
WORKFLOW_STRETCH_IMAGENET = "stretch_imagenet"
WORKFLOW_ULTRALYTICS_SEG = "ultralytics_seg"
WORKFLOWS = (WORKFLOW_LETTERBOX_IMAGENET, WORKFLOW_STRETCH_IMAGENET, WORKFLOW_ULTRALYTICS_SEG)


@dataclass(frozen=True)
class SpecimenModelSpec:
    key: str                # settings value + Hugging Face repo suffix
    label: str              # human name for logs / the GUI
    workflow: str           # one of WORKFLOWS
    imgsz: int              # the size the model was trained at; settings cannot change it
    hub_repo: str           # where the weights live
    trained_on: str         # one line: the training reference the workflow reproduces


SPECIMEN_MODELS: dict[str, SpecimenModelSpec] = {
    "unetpp_effb7_1024": SpecimenModelSpec(
        key="unetpp_effb7_1024", label="UNet++ (EfficientNet-B7)", workflow=WORKFLOW_LETTERBOX_IMAGENET,
        imgsz=1024, hub_repo="phyloforfun/lm3_specimen_segmenter__unetpp_effb7_1024",
        trained_on="letterbox_pair: INTER_AREA, white pad, ImageNet norm"),
    "birefnet_hr_swinl_1024": SpecimenModelSpec(
        key="birefnet_hr_swinl_1024", label="BiRefNet (Swin-L)", workflow=WORKFLOW_STRETCH_IMAGENET,
        imgsz=1024, hub_repo="phyloforfun/lm3_specimen_segmenter__birefnet_hr_swinl_1024",
        trained_on="path_to_image: cv2 INTER_LINEAR stretch to 1024x1024, ImageNet norm"),
    "yolo26x_seg_1280": SpecimenModelSpec(
        key="yolo26x_seg_1280", label="YOLO26x-seg", workflow=WORKFLOW_ULTRALYTICS_SEG,
        imgsz=1280, hub_repo="phyloforfun/lm3_specimen_segmenter__yolo26x_seg_1280",
        trained_on="Ultralytics LetterBox: INTER_LINEAR, pad 114 to stride multiple, /255"),
}

#: The only model the stage could run before keys existed; a settings file without a key means it.
DEFAULT_SPECIMEN_MODEL_KEY = "unetpp_effb7_1024"


class UnknownSpecimenModel(ValueError):
    """``model.key`` names no registered specimen segmenter."""


def _get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(node, key, default)


def resolve_specimen_model(stage_cfg: Any) -> tuple[SpecimenModelSpec, list[str]]:
    """Return ``(spec, warnings)`` for a ``modules.specimen_segmenter`` block.

    A missing key resolves to :data:`DEFAULT_SPECIMEN_MODEL_KEY` with a warning. (A merged
    :class:`~leafmachine3.core.config.Config` always has one: ``builtin_defaults()`` supplies the
    same default key, so the warning only fires for a raw block that bypassed the defaults.) An unknown key
    raises :class:`UnknownSpecimenModel`. A legacy ``imgsz`` that disagrees with the model's
    training size is ignored with a warning: the size belongs to the model, not the settings file.
    """
    warnings: list[str] = []
    model = _get(stage_cfg, "model", None)
    raw: Optional[str] = _get(model, "key", None)
    if raw is None or str(raw).strip() == "":
        key = DEFAULT_SPECIMEN_MODEL_KEY
        warnings.append(
            f"modules.specimen_segmenter.model.key is not set; assuming {key!r} (the UNet++ default). "
            f"Set it explicitly: one of {', '.join(SPECIMEN_MODELS)}.")
    else:
        key = str(raw).strip()
    spec = SPECIMEN_MODELS.get(key)
    if spec is None:
        raise UnknownSpecimenModel(
            f"modules.specimen_segmenter.model.key {key!r} is not a known specimen segmenter "
            f"(known: {', '.join(SPECIMEN_MODELS)})")
    legacy = _get(stage_cfg, "imgsz", None)
    if legacy not in (None, "", "auto"):
        try:
            if int(legacy) != spec.imgsz:
                warnings.append(
                    f"modules.specimen_segmenter.imgsz={legacy} is ignored: {spec.key} was trained at "
                    f"{spec.imgsz} and always runs at that size (the setting is retired).")
        except (TypeError, ValueError):
            warnings.append(f"modules.specimen_segmenter.imgsz={legacy!r} is ignored (the setting is retired).")
    return spec, warnings
