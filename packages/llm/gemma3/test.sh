#!/bin/bash

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

python3 -c '
from transformers import AutoProcessor, AutoModelForImageTextToText

device = "cpu"

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it").to(device)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/ryzers/data/toucan.jpg"},
            {"type": "text", "text": "What animal is this? Only state the name of the animal."}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
'
