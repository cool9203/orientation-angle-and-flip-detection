accelerate launch src/orientation_angle_and_flip_detection/train.py \
	`# Dataset` \
	--dataset_name /mnt/c/Users/ychsu/Downloads/沛波標記data/鋼材辨識/沛波圖形邊長定義/train_data/converted/20250716-oad-data \
	--dataset_test_split val \
	--balance \
	--image_count 100 \
	`# Model` \
	--model_name_or_path "microsoft/resnet-50" \
	`# Output` \
	--output_dir saves/resnet-50 \
	--logging_steps 10 \
	--save_strategy best \
	--load_best_model_at_end \
	--metric_for_best_model accuracy \
	`# Train` \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 4 \
	--eval_strategy epoch \
	--learning_rate 1.0e-3 \
	--num_train_epochs 10.0 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.1 \
    --ddp_timeout 180000000
