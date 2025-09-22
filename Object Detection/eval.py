from torchvision.ops import box_convert

def convert_to_coco_eval_format(predictions, image_ids):
    coco_results = []
    for preds, img_id in zip(predictions, image_ids):
        boxes = preds["boxes"].cpu()
        scores = preds["scores"].cpu()
        labels = preds["labels"].cpu()

        boxes_xywh = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh")
        for box, score, label in zip(boxes_xywh, scores, labels):
            coco_results.append({
                "image_id": int(img_id),
                "category_id": int(label),  # COCO category ID
                "bbox": [round(float(x), 2) for x in box],
                "score": round(float(score), 3)
            })
    return coco_results
