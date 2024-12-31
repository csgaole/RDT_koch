python -m scripts.koch_inference \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="checkpoints/rdt-finetune-1b-pick-and-place-3/checkpoint-150000" \
    --lang_embeddings_path="outs/pull_tissue_paper_from_bag_and_place_on_desk.pt" \
    --ctrl_freq=30 \
    --robot-path="lerobot/configs/robot/koch.yaml" \

    # --lang_embeddings_path="data/empty_lang_embed.pt" \