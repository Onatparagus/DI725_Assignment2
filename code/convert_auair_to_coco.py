import json
import os
from tqdm import tqdm

INPUT_JSON = "dataset/annotations.json"
IMG_DIR = "dataset/images"
OUTPUT_JSON = "coco_annotations/auair_coco.json"

def parse_auair():
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    categories = [{"id": i, "name": name} for i, name in enumerate(data["categories"])]
    cat_map = {i: name for i, name in enumerate(data["categories"])}
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    ann_id = 1
    for img_id, entry in enumerate(tqdm(data["annotations"])):
        file_name = entry["image_name"]
        width = int(entry["image_width:"])
        height = int(entry["image_height"])
        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        for bbox in entry["bbox"]:
            left = bbox["left"]
            top = bbox["top"]
            width_box = bbox["width"]
            height_box = bbox["height"]
            class_id = bbox["class"]

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [left, top, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0
            })
            ann_id += 1

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(coco, f, indent=4)
    print(f"Saved COCO annotations to {OUTPUT_JSON}")

if __name__ == "__main__":
    parse_auair()
