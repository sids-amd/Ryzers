# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import torch
from transformers import pipeline, AutoModel, AutoProcessor
from transformers.image_utils import load_image

# SigLIP2 model checkpoint to use
ckpt = "google/siglip2-so400m-patch16-naflex"

# Images to evaluate
images = ["/ryzers/data/toucan.jpg"]

#######################################################
######### Test zero-shot image classification #########
#######################################################

pipe = pipeline(model=ckpt, task="zero-shot-image-classification")

inputs = {
    "images": images,
    "texts": [
        "a bird",
        "a vulture",
        "a toucan",
        "two bears",
        "three bears and a bird"
    ],
}

outputs = pipe(inputs["images"], candidate_labels=inputs["texts"])

# pretty-print outputs
for i, im in enumerate(outputs):
    print("="*16+f" Image: {images[i]} "+"="*16)
    print(json.dumps(im, indent=4))

######################################
######### Test image encoder #########
######################################

model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)
for image_fpath in images:
    image = load_image(image_fpath)
    inputs = processor(images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)    

    print(f"[{image_fpath}] Embedding: {embedding} (shape: {embedding.shape})")
