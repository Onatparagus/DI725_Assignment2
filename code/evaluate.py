from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torch

class CocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, processor):
        super().__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        encoding = self.processor(images=image, annotations={"image_id": image_id, "annotations": annotations}, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target

def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}
    
def generate_predictions(model, processor, dataset, output_path, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            pixel_values, _ = dataset[idx]
            pixel_values = pixel_values.unsqueeze(0).to(device)

            outputs = model(pixel_values=pixel_values)
            results = processor.post_process_object_detection(outputs, target_sizes=[(1080, 1920)], threshold=0.5)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x_min, y_min, x_max, y_max = box.tolist()
                width = x_max - x_min
                height = y_max - y_min

                preds.append({
                    "image_id": idx,
                    "category_id": label.item(),
                    "bbox": [x_min, y_min, width, height],
                    "score": score.item()
                })

    with open(output_path, "w") as f:
        json.dump(preds, f)
    print(f"Saved predictions to {output_path}")

def run_eval(gt_path, pred_path):
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DetrForObjectDetection.from_pretrained("detr_auair_model").to(device)
    processor = DetrImageProcessor.from_pretrained("detr_auair_processor")

    train_dataset = CocoDataset("dataset/images", "coco_annotations/auair_coco.json", processor)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Run predictions
    generate_predictions(model, processor, dataset, "predictions.json", device)

    run_eval("coco_annotations/auair_coco.json", "predictions.json")
