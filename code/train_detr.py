import torch
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
import wandb
import os
from tqdm import tqdm
import json

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
        
def generate_predictions(model, processor, dataset, output_path):
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

def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}

def train():
    wandb.init(project="auair-object-detection")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=8  # Human, Car, Truck, Van, Motorbike, Bicycle, Bus, Trailer
    )

    train_dataset = CocoDataset("dataset/images", "dataset/annotations.json", processor)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(10):
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in loop:
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "detr_auair.pth")
    wandb.finish()
    generate_predictions(model, processor, train_dataset, "predictions.json")

if __name__ == "__main__":
    train()
