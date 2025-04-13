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

def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}

def train():
    wandb.init(project="auair-object-detection")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=8,  # Human, Car, Truck, Van, Motorbike, Bicycle, Bus, Trailer
        ignore_mismatched_sizes=True
    )
    print(model.class_labels_classifier)

    train_dataset = CocoDataset("dataset/images", "coco_annotations/auair_coco.json", processor)
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
    
    model.save_pretrained("detr_auair_model")
    processor.save_pretrained("detr_auair_processor")

if __name__ == "__main__":
    train()
