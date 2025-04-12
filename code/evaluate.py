from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def run_eval(gt_path, pred_path):
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    run_eval("coco_annotations/auair_coco.json", "predictions.json")
