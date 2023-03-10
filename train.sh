pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse"
output_dir="models/kurtulusbirgoren" 
save_sample_prompt="photo of kurtulusbirgoren"

accelerate launch train_dreambooth.py \
	--pretrained_model_name_or_path=$pretrained_model_name_or_path \
	--pretrained_vae_name_or_path=$pretrained_vae_name_or_path \
	--output_dir=$output_dir \
	--revision="fp16" \
	--with_prior_preservation --prior_loss_weight=1.0 \
	--seed=1337 \
	--resolution=512 \
	--train_batch_size=1 \
	--train_text_encoder \
	--mixed_precision="fp16" \
	--gradient_accumulation_steps=1 \
	--learning_rate=1e-6 \
	--lr_scheduler="constant" \
	--lr_warmup_steps=0 \
	--num_class_images=250 \
	--sample_batch_size=4 \
	--max_train_steps=1100 \
	--save_interval=10000 \
	--save_sample_prompt="$save_sample_prompt" \
	--concepts_list="concepts_list.json"
