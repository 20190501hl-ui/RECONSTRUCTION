#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多阶段重构推理模块
生成三个时间步的重构图像：50%, 80%, 100%
"""

import argparse
import os
import pickle
import torch
import torch.utils.checkpoint
from datasets import Dataset
from diffusers import StableDiffusionPipeline
from PIL import Image
from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images): 
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc

def safe_torch_load(file_path):
    """安全加载包含PIL.Image对象的torch文件"""
    try:
        # 首先尝试默认加载
        return torch.load(file_path)
    except (RuntimeError, pickle.PicklingError, Exception) as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["weights_only", "pil.image", "weightspickler", "unsupported global"]):
            # 如果是PyTorch 2.6+的安全加载问题，使用weights_only=False
            print(f"使用兼容模式加载 {file_path} (PyTorch 2.6+)")
            return torch.load(file_path, weights_only=False)
        else:
            raise e

class MultistageInference:
    def __init__(self, model_path, output_dir, device="cuda"):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device
        self.pipeline = None
        
    def load_pipeline(self):
        """加载Stable Diffusion Pipeline"""
        print(f"Loading pipeline from {self.model_path}...")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            revision=None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 加载LoRA权重（如果存在）
        if os.path.exists("output_lora"):
            try:
                self.pipeline.unet.load_attn_procs("output_lora")
                print(f"✅ Loaded LoRA weights from output_lora/")
            except Exception as e:
                print(f"⚠️ Failed to load LoRA weights: {e}")
                print("🔧 Using base Stable Diffusion model without LoRA")
        else:
            print("⚠️ LoRA weights not found, using base Stable Diffusion model")
        
        self.pipeline.to(self.device)
        print("✅ Pipeline loaded successfully!")
    
    def generate_single_stage(self, dataset, save_dir, num_images=3, inference_steps=30, seed=1337):
        """生成单阶段重构图像（基线方法）"""
        print(f"Generating single-stage images to {save_dir}...")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.manual_seed(seed)
        
        for i in range(len(dataset["text"])):
            for j in range(num_images):
                image = self.pipeline(
                    dataset["text"][i], 
                    num_inference_steps=inference_steps,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(seed + i * 100 + j)
                ).images[0]
                
                filename = f"image_{i+1:02}_{j+1:02}.jpg"
                save_path = os.path.join(save_dir, filename)
                image.save(save_path)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset['text'])} samples (single-stage)")
    
    def generate_multistage(self, dataset, save_dir, num_images=3, inference_steps=30, seed=1337):
        """生成三阶段重构图像（创新方法）"""
        print(f"Generating multistage images to {save_dir}...")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.manual_seed(seed)
        
        # 三个重构阶段：50%, 80%, 100%
        timestep_ratios = [0.5, 0.8, 1.0]
        timesteps = [int(inference_steps * ratio) for ratio in timestep_ratios]
        
        print(f"Multistage reconstruction with timesteps: {timesteps}")
        
        for i in range(len(dataset["text"])):
            prompt = dataset["text"][i]
            for stage_idx, steps in enumerate(timesteps):
                for j in range(num_images):
                    image = self.pipeline(
                        prompt,
                        num_inference_steps=steps,
                        guidance_scale=7.5,
                        generator=torch.Generator(device=self.device).manual_seed(seed + i * 100 + j)
                    ).images[0]
                    
                    # 格式: image_01_s1_01.jpg (图像ID_阶段_样本ID)
                    filename = f"image_{i+1:02}_s{stage_idx+1}_{j+1:02}.jpg"
                    save_path = os.path.join(save_dir, filename)
                    image.save(save_path)
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(dataset['text'])} samples (multistage)")

def main():
    # 内置参数配置
    config = {
        "model_path": "runwayml/stable-diffusion-v1-5",
        "lora_dir": "output_lora",
        "data_path": "data/test_dataset.pt",
        "single_save_dir": "result/single_stage_images",
        "multi_save_dir": "result/multi_stage_images", 
        "num_images": 3,
        "inference_steps": 15,
        "seed": 1337,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print("=" * 60)
    print("多阶段重构攻击 - 图像生成模块")
    print("=" * 60)
    
    # 检查数据文件
    if not os.path.exists(config["data_path"]):
        print(f"Error: Data file not found: {config['data_path']}")
        print("Please make sure the dataset is in the correct location.")
        return
    
    # 初始化推理器
    inferencer = MultistageInference(
        model_path=config["model_path"],
        output_dir=config["lora_dir"],
        device=config["device"]
    )
    
    # 加载pipeline
    try:
        inferencer.load_pipeline()
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return
    
    # 加载数据集
    print(f"Loading dataset from {config['data_path']}...")
    dataset = Dataset.from_dict(safe_torch_load(config["data_path"]))
    print(f"Loaded {len(dataset)} samples")
    
    # 生成单阶段图像（基线）
    print("\nStep 1: Generating single-stage images (baseline)...")
    inferencer.generate_single_stage(
        dataset=dataset,
        save_dir=config["single_save_dir"],
        num_images=config["num_images"],
        inference_steps=config["inference_steps"],
        seed=config["seed"]
    )
    
    # 生成多阶段图像（创新方法）
    print("\nStep 2: Generating multistage images (our method)...")
    inferencer.generate_multistage(
        dataset=dataset,
        save_dir=config["multi_save_dir"],
        num_images=config["num_images"],
        inference_steps=config["inference_steps"],
        seed=config["seed"]
    )
    
    print("\n" + "=" * 60)
    print("图像生成完成!")
    print(f"单阶段图像保存在: {config['single_save_dir']}")
    print(f"多阶段图像保存在: {config['multi_save_dir']}")
    print("接下来请运行特征提取: python extract_features.py")
    print("=" * 60)

if __name__ == "__main__":
    main()