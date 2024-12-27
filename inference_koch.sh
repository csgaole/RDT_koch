python -m scripts.koch_inference \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="checkpoints/rdt-finetune-1b-pick-and-place/checkpoint-150000" \
    --lang_embeddings_path="outs/pick_and_place_cap.pt" \
    --ctrl_freq=25 \
    --robot-path="lerobot/configs/robot/koch.yaml" \

    # --lang_embeddings_path="data/empty_lang_embed.pt" \