#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

python3 -c '
import torch
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cpu",
    torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/ryzers/data/toucan.jpg"},
            {"type": "text", "text": "What animal is this? Only state the name of the animal."}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=40)
print(output[0]["generated_text"][-1]["content"])
'
