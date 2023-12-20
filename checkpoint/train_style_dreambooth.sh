# Native Training 原生训练
# 主要用于训练风格、作画能力（需要每张图片都有对应的标签描述）
multi_gpu=0
export DATASET_NAME="toml/checkpoint/style_dreambooth.toml"   #<数据准备时创建的.toml文件>
export sample_prompts="toml/checkpoint/style_dreambooth_sample_prompts.txt"

export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3

extArgs=()
launchArgs=()

if [[ $multi_gpu == 1 ]]; then launchArgs+=("--multi_gpu"); fi
if [[ $utf8 == 1 ]]; then export PYTHONUTF8=1; fi

python -m accelerate.commands.launch ${launchArgs[@]} --num_cpu_threads_per_process=8 "sd-scripts/train_db.py" \
    --pretrained_model_name_or_path="/appdata/finetuning/sd-models/Chilloutmix-Ni-pruned-fp32-fix.safetensors"  \
    --dataset_config=$DATASET_NAME \
    --output_dir="/appdata/finetuning/output/dreambooth_model/v9"  \
    --output_name="yqRealistic_db_v9"  \
    --save_model_as=safetensors  \
    --prior_loss_weight=1.0  \
    --max_train_epochs=10  \
    --clip_skip=2 \
    --max_token_length=225  \
    --train_batch_size=4  \
    --learning_rate=2e-6 \
    --optimizer_type="AdamW8bit" \
    --lr_scheduler="cosine_with_restarts" \
    --mixed_precision="fp16" \
    --cache_latents \
    --xformers \
    --gradient_checkpointing \
    --save_precision="fp16" \
    --save_every_n_epochs 2 \
    --sample_prompts=$sample_prompts \
    --sample_sampler "euler"  \
    --sample_every_n_epochs 2 \
    --tokenizer_cache_dir="token" \
    ${extArgs[@]}

#--max_train_epochs=10  \

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