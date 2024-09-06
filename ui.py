import gradio as gr
import yaml
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

seafoam = Seafoam()

def save_config(
    name, resolution, sample_every, width, height, prompts, neg, seed, walk_seed, guidance_scale, sample_steps,
    steps, batch_size, dtype, save_every, num_repeats, cache_latents_to_disk, train_unet, train_text_encoder,
    gradient_checkpointing, gradient_accumulation_steps, optimizer, lr_scheduler, lr, low_vram,
    linear, max_step_saves_to_keep, linear_alpha, caption_ext, caption_dropout_rate, shuffle_tokens,
    use_ema, ema_decay, quantize, noise_scheduler,
    device, training_folder, folder_path, model_name_or_path, is_flux
):
    config = {
        'config': {
            'name': name,
            'process': [
                {
                    'type': 'sd_trainer',
                    'training_folder': training_folder,
                    'device': device,
                    'network': {
                        'type': 'lora',
                        'linear': linear,
                        'linear_alpha': linear_alpha
                    },
                    'save': {
                        'dtype': dtype,
                        'save_every': save_every,
                        'max_step_saves_to_keep': max_step_saves_to_keep
                    },
                    'datasets': [
                        {
                            'folder_path': folder_path,
                            'caption_ext': caption_ext,
                            'num_repeats': num_repeats,
                            'caption_dropout_rate': caption_dropout_rate,
                            'shuffle_tokens': shuffle_tokens,
                            'cache_latents_to_disk': cache_latents_to_disk,
                            'resolution': [resolution]
                        }
                    ],
                    'train': {
                        'batch_size': batch_size,
                        'steps': steps,
                        'gradient_accumulation_steps': gradient_accumulation_steps,
                        'train_unet': train_unet,
                        'train_text_encoder': train_text_encoder,
                        'gradient_checkpointing': gradient_checkpointing,
                        'noise_scheduler': noise_scheduler,
                        'optimizer': optimizer,
                        'lr_scheduler': lr_scheduler,
                        'lr': lr
                    },
                    'ema_config': {
                        'use_ema': use_ema,
                        'ema_decay': ema_decay
                    },
                    'model': {
                        'name_or_path': model_name_or_path,
                        'is_flux': is_flux,
                        'quantize': quantize,
                        'low_vram': low_vram
                    },
                    'sample': {
                        'sample_every': sample_every,
                        'width': width,
                        'height': height,
                        'prompts': prompts,
                        'neg': neg,
                        'seed': seed,
                        'walk_seed': walk_seed,
                        'guidance_scale': guidance_scale,
                        'sample_steps': sample_steps
                    }
                }
            ]
        }
    }
    
    with open(f"{name}.yaml", "w") as yaml_file:
        yaml.dump(config, yaml_file, allow_unicode=True)

    return f"Configuration saved to {name}.yaml"

with gr.Blocks(theme=seafoam) as demo:
    gr.Markdown("### 参数输入界面 (Parameter Input Interface)")
    
    # Newbie Settings
    with gr.Accordion("新手设置 (Newbie Settings)", open=True):
        with gr.Row():
            name = gr.Textbox(label="Lora名称 (name)", placeholder="my_first_flux_lora_v1")
            resolution = gr.Slider(label="训练分辨率 (resolution)", minimum=512, maximum=1536, step=1, value=1024)
        
        # Sampling (expanded)
        with gr.Accordion("生成例图采样 (Sampling)", open=True):
            with gr.Row():
                sample_every = gr.Number(label="每N步采样 (sample_every)", value=250)
                width = gr.Number(label="宽度 (width)", value=1024)
                height = gr.Number(label="高度 (height)", value=1024)


            #提示词应该改为多个值的列表，需要增减按钮，能创建多个提示次。传递给save_config时需为列表。 TODO
            prompts = gr.Textbox(label="提示词 (prompts)", placeholder="woman with red hair...", lines=5)
            #增加默认值 TODO
            neg = gr.Textbox(label="负面提示词 (neg)", placeholder="None")
            with gr.Row():
                seed = gr.Number(label="随机种子 (seed)", value=42)
                walk_seed = gr.Checkbox(label="walk种子 (walk_seed)", value=True)
                guidance_scale = gr.Slider(label="CFG (guidance_scale)", minimum=0, maximum=10, step=0.1, value=4)
                sample_steps = gr.Number(label="采样步数 (sample_steps)", value=20)
    
    # Basic Settings (collapsed)
    with gr.Accordion("基础设置 (Basic Settings)", open=False):
        with gr.Row():
            steps = gr.Number(label="训练步数 (steps)", value=4000)
            batch_size = gr.Number(label="并行训练数 (batch_size)", value=1)
            # dtype完善列表 TODO
            dtype = gr.Dropdown(label="保存精度 (dtype)", choices=["float16", "bf16"], value="float16")
        with gr.Row():
            save_every = gr.Number(label="每N步保存 (save_every)", value=250)
            num_repeats = gr.Number(label="重复次数 (num_repeats)", value=20)
            cache_latents_to_disk = gr.Checkbox(label="缓存到磁盘 (cache_latents_to_disk)", value=True)
        with gr.Row():
            train_unet = gr.Checkbox(label="训练Unet (train_unet)", value=True)
            train_text_encoder = gr.Checkbox(label="训练文本编码器 (train_text_encoder)", value=False)
            gradient_checkpointing = gr.Checkbox(label="梯度检查点 (gradient_checkpointing)", value=True)
        with gr.Row():
            gradient_accumulation_steps = gr.Number(label="梯度累积步数 (gradient_accumulation_steps)", value=1)
            # optimizer完善列表 TODO
            optimizer = gr.Textbox(label="优化器 (optimizer)", value="adamw8bit")
            # lr_scheduler完善列表 TODO
            lr_scheduler = gr.Textbox(label="学习率调度器 (lr_scheduler)", value="cosine")
        with gr.Row():
            lr = gr.Number(label="学习率 (lr)", value=4e-4)
            low_vram = gr.Checkbox(label="低显存模式 (low_vram)", value=True)
    """
    TODO
    增加可选参数(需要一个是否启用的方框，还需要一个值，在同一行)
    performance_log_every, 默认值为1000,不启用
    trigger_word, 默认值为"p3r5on",，不启用
    skip_first_sample,无值,默认不启用
    """

    # Advanced Settings (collapsed)
    with gr.Accordion("高级设置 (Advanced Settings)", open=False):
        with gr.Row():
            
            linear = gr.Slider(label="线性 (linear)", minimum=1, maximum=64, step=1, value=16)
            max_step_saves_to_keep = gr.Number(label="保存的最大步数 (max_step_saves_to_keep)", value=12)
            linear_alpha = gr.Slider(label="线性Alpha (linear_alpha)", minimum=1, maximum=64, step=1, value=16)
        with gr.Row():
            caption_ext = gr.Textbox(label="打标文件格式 (caption_ext)", value="txt")
            caption_dropout_rate = gr.Slider(label="关键字删除率 (caption_dropout_rate)", minimum=0.0, maximum=0.1, step=0.01, value=0)
            shuffle_tokens = gr.Checkbox(label="打乱提示词 (shuffle_tokens)", value=False)
        with gr.Row():
            use_ema = gr.Checkbox(label="使用EMA (use_ema)", value=True)
            ema_decay = gr.Slider(label="EMA衰减 (ema_decay)", minimum=0.9, maximum=1.0, step=0.01, value=0.99)
            quantize = gr.Checkbox(label="量化 (quantize)", value=True)
            # noise_scheduler完善列表 TODO
            noise_scheduler = gr.Textbox(label="噪声调度器 (noise_scheduler)", value="flowmatch")
        
        # Locked Settings (collapsed)
        with gr.Accordion("锁定设置 (Locked Settings)", open=False):
            with gr.Row():
            
                device = gr.Textbox(label="设备 (device)", value="cuda:0", interactive=False)
                training_folder = gr.Textbox(label="训练文件夹 (training_folder)", value="/usrdata/train_output_flux_aitoolkit", interactive=False)
                folder_path = gr.Textbox(label="数据集路径 (folder_path)", value="/usrdata/train_pics_flux_aitoolkit", interactive=False)
            with gr.Row():
                model_name_or_path = gr.Textbox(label="模型路径 (model_name_or_path)", value="/chenyudata/ComfyUI/models/diffusers/FLUX.1-dev", interactive=False)
                is_flux = gr.Checkbox(label="使用Flux (is_flux)", value=True, interactive=False)


    
    submit_btn = gr.Button("保存配置 (Save Configuration)")
    output = gr.Textbox()
    
    submit_btn.click(
        save_config, 
        [
            name, resolution, sample_every, width, height, prompts, neg, seed, walk_seed, guidance_scale, sample_steps,
            steps, batch_size, dtype, save_every, num_repeats, cache_latents_to_disk, train_unet, train_text_encoder,
            gradient_checkpointing, gradient_accumulation_steps, optimizer, lr_scheduler, lr, low_vram,
            linear, max_step_saves_to_keep, linear_alpha, caption_ext, caption_dropout_rate, shuffle_tokens,
            use_ema, ema_decay, quantize, noise_scheduler,
            device, training_folder, folder_path, model_name_or_path, is_flux
        ], 
        output
    )

#demo.launch(server_name="0.0.0.0",server_port=8188)
demo.launch(server_name="0.0.0.0",server_port=7860)
