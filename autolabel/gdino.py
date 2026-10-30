"""
gdino.py - run Grounding DINO (open-vocabulary detection) on images.

How it works, in one paragraph
------------------------------
All class descriptions go into ONE text prompt, lowercase, each ending with
a dot: "a worker wearing a hard hat, seen from above. a long narrow ...".
The model proposes 900 boxes and, for each box, a score for every TEXT TOKEN.
We give each box one score per class = its best token score inside that
class's phrase, then label the box with its best class. Same per-class NMS
as our OWLv2 code. We compute the class scores ourselves instead of using
the processor's phrase matching, so labels always map back to our class
names. Boxes come out relative to the original image (no padding).

Model ids verified on Hugging Face 2026-10-30: IDEA-Research/grounding-dino-tiny, -base.
"""

from __future__ import annotations

from autolabel.owlv2 import Detection, pick_device

DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"


def build_prompt(descriptions: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """One prompt string and the (start, end) character span of each class phrase in it."""
    parts, spans, pos = [], [], 0
    for d in descriptions:
        phrase = d.strip().lower().rstrip(".")
        spans.append((pos, pos + len(phrase)))
        parts.append(phrase + ".")
        pos += len(phrase) + 2  # ". "
    return " ".join(parts), spans


def token_owner(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    """Class index per token (-1 for special tokens, dots and anything outside a phrase)."""
    owner = []
    for s, e in offsets:
        cls = -1
        if e > s:
            for c, (a, b) in enumerate(spans):
                if s >= a and e <= b:
                    cls = c
                    break
        owner.append(cls)
    return owner


class GroundingDinoDetector:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto", nms_iou: float = 0.5):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = pick_device(device)
        self.nms_iou = nms_iou
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self._prompt_cache: dict = {}

    def _prompt(self, descriptions: list[str]):
        key = tuple(descriptions)
        if key not in self._prompt_cache:
            text, spans = build_prompt(descriptions)
            tok = self.processor.tokenizer(text, return_offsets_mapping=True)
            self._prompt_cache[key] = (text, tok["input_ids"], token_owner(tok["offset_mapping"], spans))
        return self._prompt_cache[key]

    def detect(self, image, descriptions: list[str], threshold: float = 0.05) -> list[Detection]:
        from torchvision.ops import batched_nms

        torch = self.torch
        width, height = image.size
        text, ids, owner = self._prompt(descriptions)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        if inputs["input_ids"][0].tolist() != ids:
            raise RuntimeError("Processor tokenised the prompt differently than expected; class mapping would be wrong")
        with torch.inference_mode():
            out = self.model(**inputs)

        probs = torch.sigmoid(out.logits[0].float())  # [900, 256 text positions]
        n_tok = len(owner)
        cls_scores = torch.full((probs.shape[0], len(descriptions)), 0.0, device=probs.device)
        owner_t = torch.tensor(owner, device=probs.device)
        for c in range(len(descriptions)):
            cols = (owner_t == c).nonzero().flatten()
            if len(cols):
                cls_scores[:, c] = probs[:, :n_tok][:, cols].max(dim=1).values
        scores, labels = cls_scores.max(dim=1)

        keep = scores >= threshold
        if not keep.any():
            return []
        cx, cy, bw, bh = out.pred_boxes[0].float()[keep].unbind(-1)
        xyxy = torch.stack([(cx - bw / 2) * width, (cy - bh / 2) * height,
                            (cx + bw / 2) * width, (cy + bh / 2) * height], -1)
        sc, lb = scores[keep], labels[keep]
        kept = batched_nms(xyxy, sc, lb, self.nms_iou)

        dets = []
        for i in kept.tolist():
            x1, y1, x2, y2 = xyxy[i].tolist()
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(Detection(int(lb[i]), float(sc[i]), (x1, y1, x2 - x1, y2 - y1)))
        return dets

    def peak_vram_gb(self) -> float | None:
        if self.device.startswith("cuda"):
            return self.torch.cuda.max_memory_allocated() / 1024**3
        return None
