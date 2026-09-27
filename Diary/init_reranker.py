#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Alibaba-NLP/gte-reranker-modernbert-base"

print("Loading ModernBERT reranker...")

tokenizer = AutoTokenizer.from_pretrained(
	MODEL_NAME,
	local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

model = model.to("cuda")
model.eval()

print("ModernBERT loaded.")