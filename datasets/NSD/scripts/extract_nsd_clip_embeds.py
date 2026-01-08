import json
from pathlib import Path

import torch
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

ROOT = Path(__file__).parents[1]
NSD_ROOT = ROOT / "data/NSD"


class NSDImageDataset(Dataset):
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        self.num_images = 73000

    def __getitem__(self, idx):
        return Image.fromarray(self.file["imgBrick"][idx])

    def __len__(self):
        return self.num_images


def collate(batch):
    return batch


def main():
    device = torch.device("cuda")
    dataset = NSDImageDataset(NSD_ROOT / "nsd_stimuli.hdf5")

    loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=8,
        drop_last=False,
        shuffle=False,
        collate_fn=collate,
    )

    with open(ROOT / "metadata/coco_categories.json") as f:
        coco_categories = json.load(f)
    cluster_labels = [f"photo of {label}" for label in coco_categories]
    print("\n".join(cluster_labels[:10]))

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device)

    embeds = []
    logits = []
    for batch in tqdm(loader):
        inputs = processor(text=cluster_labels, images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)

        logit = outputs["logits_per_image"].cpu().numpy()
        embed = outputs["image_embeds"].cpu().numpy()
        logits.append(logit)
        embeds.append(embed)

    logits = np.concatenate(logits)
    embeds = np.concatenate(embeds)

    np.save(ROOT / "data/nsd_clip_embeds.npy", embeds)
    np.save(ROOT / "data/nsd_clip_coco_logits.npy", logits)


if __name__ == "__main__":
    main()
