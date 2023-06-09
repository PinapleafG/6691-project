CUDA_VISIBLE_DEVICES=2 python train.py \
    --model_name 'feature_contrast' \
    --train_img_path 'single_object/data/train_img' \
    --train_mask_path 'single_object/data/train_mask' \
    --valid_img_path 'single_object/data/valid_img' \
    --valid_mask_path 'single_object/data/valid_mask' \
    --description_path 'gpt3.5_output.json' \
    --img_size 224 \
    --num_classes 2 \
    --epochs 200 \
    --base_lr 0.001 \
    --dropout 0 \
    --batch_size 64 \
    --embed_dim 96 \
    --depths_encoder 2 2 2 2 \
    --depths_decoder 1 2 2 2\
    --num_heads 3 6 12 24 \
    --contrast "feature"