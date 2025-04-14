from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from math import isnan

def evaluate_per_class(gt_file, pred_file):
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()

    cat_ids = coco_gt.getCatIds()
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.loadCats(cat_ids)}

    precisions = coco_eval.eval['precision']  # [TxRxKxAxM]
    results = []

    print("\nPer-class Average Precision @ IoU=0.50:0.95")
    print("--------------------------------------------------")
    for idx, catId in enumerate(cat_ids):
        precision = precisions[:, :, idx, 0, -1]  # IoU, area=all, maxDet=100
        precision = precision[precision > -1]
        ap = precision.mean() if precision.size else float('nan')
        results.append(ap * 100)
        print(f"{cat_id_to_name[catId]:<10}: {ap * 100:.2f}")

    valid_aps = [r for r in results if not isnan(r)]
    mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
    print("--------------------------------------------------")
    print(f"{'mAP':<10}: {mAP:.2f}")
    return results

evaluate_per_class("coco_annotations/auair_coco.json", "predictions.json")
