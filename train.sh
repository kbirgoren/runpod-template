pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse"
output_dir="trained" 
revision="fp16"
prior_loss_weight=1.0
seed=1337
resolution=512
train_batch_size=1
mixed_precision="fp16"
gradient_accumulation_steps=1
learning_rate=1e-6
lr_scheduler="constant"
lr_warmup_steps=0
num_class_images=50
sample_batch_size=4
max_train_steps=1100
save_interval=10000
save_sample_prompt="photo of kurtulusbirgoren"
concepts_list="concepts_list.json"

accelerate launch train_dreambooth.py --pretrained_model_name_or_path=$pretrained_model_name_or_path --pretrained_vae_name_or_path=$pretrained_vae_name_or_path --output_dir=$output_dir --revision=$revision  --with_prior_preservation --prior_loss_weight=$prior_loss_weight --seed=$seed --resolution=$resolution --train_batch_size=$train_batch_size --train_text_encoder --use_8bit_adam --mixed_precision=$mixed_precision --gradient_accumulation_steps=$gradient_accumulation_steps --learning_rate=$learning_rate --lr_scheduler=$lr_scheduler --lr_warmup_steps=$lr_warmup_steps --num_class_images=$num_class_images --sample_batch_size=$sample_batch_size --max_train_steps=$max_train_steps --save_interval=$save_interval --save_sample_prompt=$save_sample_prompt --not_cache_latents --concepts_list=$concepts_list
