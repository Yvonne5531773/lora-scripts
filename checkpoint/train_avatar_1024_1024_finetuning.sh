# Native Training 原生训练
# 主要用于训练风格、作画能力（需要每张图片都有对应的标签描述）
multi_gpu=1
export DATASET_NAME="toml/checkpoint/avatar_1024_1024_finetuning.toml"   #<数据准备时创建的.toml文件>
export sample_prompts="toml/checkpoint/avatar_1024_1024_finetuning_sample_prompts.txt"

export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3

extArgs=()
launchArgs=()

if [[ $multi_gpu == 1 ]]; then launchArgs+=("--multi_gpu"); fi
if [[ $utf8 == 1 ]]; then export PYTHONUTF8=1; fi

#launchArgs+=("--config_file huggingface/accelerate/single_machine_config.yaml")

#Chilloutmix-Ni-pruned-fp32-fix.safetensors
#nai_model.ckpt
#v1-5-pruned-emaonly.safetensors

python -m accelerate.commands.launch ${launchArgs[@]} --num_cpu_threads_per_process=8 "sd-scripts/fine_tune.py" \
    --pretrained_model_name_or_path="/appdata/finetuning/sd-models/Chilloutmix-Ni-pruned-fp32-fix.safetensors"  \
    --dataset_config=$DATASET_NAME \
    --output_dir="/appdata/finetuning/output/finetuning_avatar/v19"  \
    --output_name="yqAvatar_ft_v19"  \
    --save_model_as=safetensors  \
    --max_train_epochs=15 \
    --clip_skip=2 \
    --max_token_length=225  \
    --train_batch_size=4  \
    --learning_rate=1e-6 \
    --use_8bit_adam \
    --lr_scheduler="cosine_with_restarts" \
    --mixed_precision="fp16" \
    --xformers \
    --gradient_checkpointing \
    --save_precision="fp16" \
    --save_every_n_epochs 5 \
    --sample_prompts=$sample_prompts \
    --sample_sampler "euler"  \
    --sample_every_n_epochs 5 \
    --tokenizer_cache_dir="token" \
    ${extArgs[@]}

#usage: fine_tune.py [-h] [--v2] [--v_parameterization] [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH] [--tokenizer_cache_dir TOKENIZER_CACHE_DIR]
#                    [--train_data_dir TRAIN_DATA_DIR] [--shuffle_caption] [--caption_extension CAPTION_EXTENSION] [--caption_extention CAPTION_EXTENTION]
#                    [--keep_tokens KEEP_TOKENS] [--caption_prefix CAPTION_PREFIX] [--caption_suffix CAPTION_SUFFIX] [--color_aug] [--flip_aug]
#                    [--face_crop_aug_range FACE_CROP_AUG_RANGE] [--random_crop] [--debug_dataset] [--resolution RESOLUTION] [--cache_latents]
#                    [--vae_batch_size VAE_BATCH_SIZE] [--cache_latents_to_disk] [--enable_bucket] [--min_bucket_reso MIN_BUCKET_RESO]
#                    [--max_bucket_reso MAX_BUCKET_RESO] [--bucket_reso_steps BUCKET_RESO_STEPS] [--bucket_no_upscale] [--token_warmup_min TOKEN_WARMUP_MIN]
#                    [--token_warmup_step TOKEN_WARMUP_STEP] [--dataset_class DATASET_CLASS] [--caption_dropout_rate CAPTION_DROPOUT_RATE]
#                    [--caption_dropout_every_n_epochs CAPTION_DROPOUT_EVERY_N_EPOCHS] [--caption_tag_dropout_rate CAPTION_TAG_DROPOUT_RATE] [--in_json IN_JSON]
#                    [--dataset_repeats DATASET_REPEATS] [--output_dir OUTPUT_DIR] [--output_name OUTPUT_NAME] [--huggingface_repo_id HUGGINGFACE_REPO_ID]
#                    [--huggingface_repo_type HUGGINGFACE_REPO_TYPE] [--huggingface_path_in_repo HUGGINGFACE_PATH_IN_REPO] [--huggingface_token HUGGINGFACE_TOKEN]
#                    [--huggingface_repo_visibility HUGGINGFACE_REPO_VISIBILITY] [--save_state_to_huggingface] [--resume_from_huggingface] [--async_upload]
#                    [--save_precision {None,float,fp16,bf16}] [--save_every_n_epochs SAVE_EVERY_N_EPOCHS] [--save_every_n_steps SAVE_EVERY_N_STEPS]
#                    [--save_n_epoch_ratio SAVE_N_EPOCH_RATIO] [--save_last_n_epochs SAVE_LAST_N_EPOCHS] [--save_last_n_epochs_state SAVE_LAST_N_EPOCHS_STATE]
#                    [--save_last_n_steps SAVE_LAST_N_STEPS] [--save_last_n_steps_state SAVE_LAST_N_STEPS_STATE] [--save_state] [--resume RESUME]
#                    [--train_batch_size TRAIN_BATCH_SIZE] [--max_token_length {None,150,225}] [--mem_eff_attn] [--xformers] [--sdpa] [--vae VAE]
#                    [--max_train_steps MAX_TRAIN_STEPS] [--max_train_epochs MAX_TRAIN_EPOCHS] [--max_data_loader_n_workers MAX_DATA_LOADER_N_WORKERS]
#                    [--persistent_data_loader_workers] [--seed SEED] [--gradient_checkpointing] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
#                    [--mixed_precision {no,fp16,bf16}] [--full_fp16] [--full_bf16] [--clip_skip CLIP_SKIP] [--logging_dir LOGGING_DIR]
#                    [--log_with {tensorboard,wandb,all}] [--log_prefix LOG_PREFIX] [--log_tracker_name LOG_TRACKER_NAME] [--log_tracker_config LOG_TRACKER_CONFIG]
#                    [--wandb_api_key WANDB_API_KEY] [--noise_offset NOISE_OFFSET] [--multires_noise_iterations MULTIRES_NOISE_ITERATIONS]
#                    [--ip_noise_gamma IP_NOISE_GAMMA] [--multires_noise_discount MULTIRES_NOISE_DISCOUNT] [--adaptive_noise_scale ADAPTIVE_NOISE_SCALE]
#                    [--zero_terminal_snr] [--min_timestep MIN_TIMESTEP] [--max_timestep MAX_TIMESTEP] [--lowram] [--sample_every_n_steps SAMPLE_EVERY_N_STEPS]
#                    [--sample_every_n_epochs SAMPLE_EVERY_N_EPOCHS] [--sample_prompts SAMPLE_PROMPTS]
#                    [--sample_sampler {ddim,pndm,lms,euler,euler_a,heun,dpm_2,dpm_2_a,dpmsolver,dpmsolver++,dpmsingle,k_lms,k_euler,k_euler_a,k_dpm_2,k_dpm_2_a}]
#                    [--config_file CONFIG_FILE] [--output_config] [--metadata_title METADATA_TITLE] [--metadata_author METADATA_AUTHOR]
#                    [--metadata_description METADATA_DESCRIPTION] [--metadata_license METADATA_LICENSE] [--metadata_tags METADATA_TAGS]
#                    [--save_model_as {None,ckpt,safetensors,diffusers,diffusers_safetensors}] [--use_safetensors] [--optimizer_type OPTIMIZER_TYPE]
#                    [--use_8bit_adam] [--use_lion_optimizer] [--learning_rate LEARNING_RATE] [--max_grad_norm MAX_GRAD_NORM]
#                    [--optimizer_args [OPTIMIZER_ARGS ...]] [--lr_scheduler_type LR_SCHEDULER_TYPE] [--lr_scheduler_args [LR_SCHEDULER_ARGS ...]]
#                    [--lr_scheduler LR_SCHEDULER] [--lr_warmup_steps LR_WARMUP_STEPS] [--lr_scheduler_num_cycles LR_SCHEDULER_NUM_CYCLES]
#                    [--lr_scheduler_power LR_SCHEDULER_POWER] [--dataset_config DATASET_CONFIG] [--min_snr_gamma MIN_SNR_GAMMA]
#                    [--scale_v_pred_loss_like_noise_pred] [--v_pred_like_loss V_PRED_LIKE_LOSS] [--weighted_captions] [--diffusers_xformers]
#                    [--train_text_encoder]

# 以下是核心参数介绍：
# 主要的几个
# --train_text_encoder 训练文本编码器
# --mixed_precision="fp16" 混合精度训练
# - center_crop
# 是否裁剪图片，一般如果你的数据集不是正方形的话，需要裁剪
# - resolution
# 图片的分辨率，一般是512，使用该参数会自动缩放输入图像
# 可以配合center_crop使用，达到裁剪成正方形并缩放到512*512的效果
# - instance_prompt
# 如果你希望训练的是特定的人物，使用该参数
# 如 --instance_prompt="a photo of <xxx> girl"
# - use_txt_as_label
# 是否读取与图片同名的txt文件作为label
# 如果你要训练的是整个大模型的图像风格，那么可以使用该参数
# 该选项会忽略instance_prompt参数传入的内容
# - learning_rate
# 学习率，一般是2e-6，是训练中需要调整的关键参数
# 太大会导致模型不收敛，太小的话，训练速度会变慢
# - max_train_steps
# 训练的最大步数，一般是1000，如果你的数据集比较大，那么可以适当增大该值
# - save_model_every_n_steps
# 每多少步保存一次模型，方便查看中间训练的结果找出最优的模型，也可以用于断点续训

#enable_bucket	true	o	o		开启 bucket 来支持不同长宽比的训练图片
#resolution	256, [512, 512]	o	o		训练时的图片分辨率
#flip_aug	true	o	o	o	水平翻转数据增强，要求训练目标对左右方向不敏感
#random_crop	false	o	o	o	随机裁剪数据增强
#color_aug	false	o	o	o	颜色数据增强，要求训练目标对颜色不敏感
#shuffle_caption	true	o	o	o	打乱文本描述
#keep_tokens	2	o	o	o	保持前多少个 token 顺序不被打乱
#num_repeats	10	o	o	o	每张图片在一个 epoch 内重复多少次