"""
owlv2.py - run OWLv2 (open-vocabulary object detection) on images.

How OWLv2 works, in one paragraph
---------------------------------
OWLv2 looks at an image and a list of text prompts ("a person", "a yellow
lifting platform"). It cuts the image into a grid of patches (60x60 = 3600
for the base model) and, for every patch, predicts one box plus a score for
how well that box matches each prompt. We keep each box's best-matching
prompt, drop low scores, then remove near-duplicate boxes of the same class
(non-maximum suppression, "NMS"), because neighbouring patches often predict
almost the same box.

Checked against transformers 5.22.1 (Owlv2Processor / Owlv2ForObjectDetection).
OWLv2 pads every image to a square before running; post-processing scales
boxes back using the longer image side, so we pass the real (height, width).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"  # verified on Hugging Face, 620 MB

# Image-guided queries ("find more things like this example box")
# ----------------------------------------------------------------
# For each example we run the model on the example's whole image, find the
# model's own predicted box that overlaps the example box best, and keep that
# box's class embedding as the query (the approach of the OWL-ViT paper).
# For a new image we compare every patch with every query, keep each class's
# best match over its examples, and turn it into a 0..1 score with a sigmoid.
# We do NOT use processor.post_process_image_guided_detection: in
# transformers 5.20.2 it rescales scores per image so the best box is always
# 1.0, even when the object isn't there (verified in the source).


@dataclass
class Detection:
    label: int  # index into the prompt list
    score: float
    bbox: tuple[float, float, float, float]  # x, y, w, h in pixels


def xywh_to_rel_xyxy(bbox, size: float) -> list[float]:
    """Pixel [x, y, w, h] -> [x1, y1, x2, y2] relative to the padded square (side = longer image side)."""
    x, y, w, h = bbox
    return [x / size, y / size, (x + w) / size, (y + h) / size]


def best_box_index(pred_cxcywh, target_xyxy) -> tuple[int, float]:
    """Index (and IoU) of the predicted box (cx, cy, w, h, relative) that overlaps target_xyxy best."""
    import torch
    from torchvision.ops import box_convert, box_iou

    pred = box_convert(torch.as_tensor(pred_cxcywh, dtype=torch.float32), "cxcywh", "xyxy")
    ious = box_iou(pred, torch.tensor([target_xyxy], dtype=torch.float32))[:, 0]
    i = int(ious.argmax())
    return i, float(ious[i])


def class_max(logits, owner: list[int], n_classes: int):
    """logits [patches, queries] -> [patches, classes]: each class keeps its best query (example)."""
    import torch

    owner_t = torch.as_tensor(owner, device=logits.device)
    out = torch.full((logits.shape[0], n_classes), float("-inf"), device=logits.device, dtype=logits.dtype)
    for c in range(n_classes):
        cols = (owner_t == c).nonzero().flatten()
        if len(cols):
            out[:, c] = logits[:, cols].max(dim=1).values
    return out


def pick_device(prefer: str = "auto") -> str:
    """'cuda' if a working GPU is available, otherwise 'cpu'."""
    import torch

    if prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


class Owlv2Detector:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto", nms_iou: float = 0.5):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = pick_device(device)
        self.nms_iou = nms_iou
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def detect(self, image, prompts: list[str], threshold: float = 0.05) -> list[Detection]:
        """
        image: a PIL image (RGB). prompts: one text description per class.
        Returns detections with score >= threshold, duplicates removed per class.
        """
        from torchvision.ops import batched_nms

        torch = self.torch
        width, height = image.size

        inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        result = self.processor.post_process_grounded_object_detection(
            outputs, threshold=threshold, target_sizes=[(height, width)]
        )[0]
        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
        if len(scores) == 0:
            return []

        keep = batched_nms(boxes.float(), scores.float(), labels, self.nms_iou)

        dets = []
        for i in keep.tolist():
            x1, y1, x2, y2 = boxes[i].tolist()
            # Clip to the real image: the padded area can produce boxes past the edge.
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(Detection(int(labels[i]), float(scores[i]), (x1, y1, x2 - x1, y2 - y1)))
        return dets

    def _image_features(self, image):
        """Patch features and predicted boxes (cx, cy, w, h relative to the padded square) for one image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            feature_map, _ = self.model.image_embedder(pixel_values=inputs["pixel_values"])
            b, h, w, d = feature_map.shape
            feats = feature_map.reshape(b, h * w, d)
            boxes = self.model.box_predictor(feats, feature_map)
        return feats, boxes[0]

    def query_embedding(self, image, bbox) -> tuple[object, float]:
        """Class embedding of the model's box that best overlaps bbox ([x, y, w, h] pixels) in image."""
        feats, boxes = self._image_features(image)
        with self.torch.inference_mode():
            _, class_embeds = self.model.class_predictor(feats)
        i, overlap = best_box_index(boxes.float().cpu(), xywh_to_rel_xyxy(bbox, max(image.size)))
        return class_embeds[0, i], overlap

    def detect_image_guided(self, image, queries: list[list], threshold: float = 0.05) -> list[Detection]:
        """
        queries[c] = list of query embeddings for class c (from query_embedding).
        Every class is scored on its own, so one object can get boxes of two classes.
        """
        from torchvision.ops import batched_nms

        torch = self.torch
        width, height = image.size
        size = float(max(width, height))
        feats, boxes = self._image_features(image)

        flat = [q for qs in queries for q in qs]
        owner = [c for c, qs in enumerate(queries) for _ in qs]
        q = torch.stack(flat)[None].to(self.device)
        with torch.inference_mode():
            logits, _ = self.model.class_predictor(feats, query_embeds=q, query_mask=torch.ones(q.shape[:2], device=self.device))
        scores = torch.sigmoid(class_max(logits[0].float(), owner, len(queries)))  # [patches, classes]

        keep = (scores >= threshold).nonzero()
        if len(keep) == 0:
            return []
        patch, label = keep[:, 0], keep[:, 1]
        cx, cy, bw, bh = boxes[patch].float().unbind(-1)
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * size
        sc = scores[patch, label]
        kept = batched_nms(xyxy, sc, label, self.nms_iou)

        dets = []
        for i in kept.tolist():
            x1, y1, x2, y2 = xyxy[i].tolist()
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(Detection(int(label[i]), float(sc[i]), (x1, y1, x2 - x1, y2 - y1)))
        return dets

    def box_features(self, image) -> tuple[object, object, object]:
        """
        Examples mode (CONTRACT #9, owlv2 rules): one vision pass, every predicted box of the image.
        Returns on the CPU, float32: class embeddings [boxes, 512] L2-normalized, boxes [boxes, 4] as
        cx, cy, w, h relative to the padded square (side = longer image side), objectness [boxes] (sigmoid).
        """
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            feature_map, _ = self.model.image_embedder(pixel_values=inputs["pixel_values"])
            b, h, w, d = feature_map.shape
            feats = feature_map.reshape(b, h * w, d)
            _, class_embeds = self.model.class_predictor(feats)       # returned un-normalized
            boxes = self.model.box_predictor(feats, feature_map)
            objectness = self.model.objectness_predictor(feats).sigmoid()
        embeds = torch.nn.functional.normalize(class_embeds[0].float(), dim=-1)
        return embeds.cpu(), boxes[0].float().cpu(), objectness[0].float().cpu()

    def peak_vram_gb(self) -> float | None:
        if self.device.startswith("cuda"):
            return self.torch.cuda.max_memory_allocated() / 1024**3
        return None
