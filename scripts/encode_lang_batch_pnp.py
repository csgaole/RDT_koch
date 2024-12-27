import os
import json

import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder


GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
# Modify the TARGET_DIR to your dataset path
# TARGET_DIR = "data/datasets/agilex/tfrecords/"

SAVE_DIR = "outs/"


def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    
    instructions = ["Pick up the gray cap on the right and put it into the packaging box on the left."]

    # Encode the instructions
    tokenized_res = tokenizer(
        instructions, return_tensors="pt",
        padding="longest",
        truncation=True
    )
    tokens = tokenized_res["input_ids"].to(device)
    attn_mask = tokenized_res["attention_mask"].to(device)
    
    with torch.no_grad():
        text_embeds = text_encoder(
            input_ids=tokens,
            attention_mask=attn_mask
        )["last_hidden_state"].detach().cpu()
    
    attn_mask = attn_mask.cpu().bool()

    # Save the embeddings for training use
    for i in range(len(instructions)):
        text_embed = text_embeds[i][attn_mask[i]]
        save_path = os.path.join(SAVE_DIR, f"lang_embed_{i}.pt")
        torch.save(text_embed, save_path)

if __name__ == "__main__":
    main()
