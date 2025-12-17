#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多阶段特征提取模块
从三阶段重构图像中提取6维特征向量
"""

import os
import pickle
import torch
import numpy as np
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import DeiTFeatureExtractor, DeiTModel, AutoImageProcessor, BeitModel, ViTModel

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

class MultiStageFeatureExtractor:
    def __init__(self, image_encoder="deit", method="cosine", device="cuda"):
        self.image_encoder = image_encoder
        self.method = method
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.feature_extractor = None
        self.model = None
        
    def load_model(self):
        """加载图像编码模型"""
        print(f"尝试加载 {self.image_encoder} 模型...")
        
        try:
            if self.image_encoder == "deit":
                self.feature_extractor = DeiTFeatureExtractor.from_pretrained(
                    "facebook/deit-base-distilled-patch16-384"
                )
                self.model = DeiTModel.from_pretrained(
                    "facebook/deit-base-distilled-patch16-384", 
                    add_pooling_layer=False
                )
            elif self.image_encoder == "beit":
                self.feature_extractor = AutoImageProcessor.from_pretrained(
                    "microsoft/beit-base-patch16-224-pt22k"
                )
                self.model = BeitModel.from_pretrained(
                    "microsoft/beit-base-patch16-224-pt22k"
                )
            elif self.image_encoder == "vit":
                self.feature_extractor = AutoImageProcessor.from_pretrained(
                    "google/vit-base-patch16-224-in21k"
                )
                self.model = ViTModel.from_pretrained(
                    "google/vit-base-patch16-224-in21k"
                )
            else:
                raise ValueError(f"Unsupported encoder: {self.image_encoder}")
            
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式以提高推理速度
            
            # 启用混合精度推理
            if self.device.type == "cuda":
                self.model.half()
                print(f"Model loaded on {self.device} with half precision")
            else:
                print(f"Model loaded on {self.device}")
        
        except Exception as e:
            error_msg = str(e).lower()
            if "couldn't connect" in error_msg or "huggingface.co" in error_msg:
                print(f"[WARNING] HuggingFace模型下载失败: {e}")
                print("[INFO] 切换到离线特征提取模式...")
                
                # 使用离线特征提取器
                from offline_extractor import OfflineFeatureExtractor
                self.offline_extractor = OfflineFeatureExtractor(method="advanced", device=str(self.device))
                self.use_offline = True
                return
            else:
                raise e
        
        self.use_offline = False
    
    def compute_distance(self, emb1, emb2):
        """计算两个嵌入向量之间的距离"""
        if self.method == "cosine":
            similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
            # 转换为距离（1 - 相似度）
            distance = 1 - similarity
        elif self.method == "euclidean":
            distance = torch.nn.functional.pairwise_distance(emb1, emb2, dim=1)
        else:
            raise ValueError(f"Unsupported distance method: {self.method}")
        
        return distance.cpu().detach().numpy()
    
    def extract_single_stage_features(self, dataset, sample_dir, num_samples=3, batch_size=8):
        """提取单阶段特征（基线方法）- 批量优化版本"""
        print("Extracting single-stage features...")
        
        # 如果使用离线模式
        if hasattr(self, 'use_offline') and self.use_offline:
            return self.offline_extractor.extract_single_stage_features(dataset, sample_dir, num_samples)
        
        features = []
        
        # 批量处理原始图像
        original_images = [dataset["image"][i].convert("RGB") for i in range(len(dataset["image"]))]
        
        # 分批处理原始图像特征提取
        original_embeddings = []
        for batch_start in tqdm(range(0, len(original_images), batch_size), desc="Extracting original features"):
            batch_end = min(batch_start + batch_size, len(original_images))
            batch_images = original_images[batch_start:batch_end]
            
            # 批量处理
            inputs = self.feature_extractor(batch_images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # 转换为半精度以匹配模型
            if self.device.type == "cuda":
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].half()
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state
            
            for i in range(len(batch_images)):
                original_embeddings.append(batch_embeddings[i:i+1])
        
        # 处理重构图像
        for i in tqdm(range(len(dataset["image"])), desc="Processing reconstructed images"):
            original_embedding = original_embeddings[i]
            
            # 动态分配标签：前50%为成员(1)，后50%为非成员(0)
            membership = 1 if i < len(dataset["image"]) // 2 else 0
            
            # 批量加载重构图像
            recon_images = []
            valid_indices = []
            for j in range(num_samples):
                filename = f"image_{i+1:02}_{j+1:02}.jpg"
                image_path = os.path.join(sample_dir, filename)
                
                if os.path.exists(image_path):
                    recon_image = Image.open(image_path).convert("RGB")
                    recon_images.append(recon_image)
                    valid_indices.append(j)
            
            if recon_images:
                # 批量处理重构图像
                inputs = self.feature_extractor(recon_images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
                # 转换为半精度以匹配模型
                if self.device.type == "cuda":
                    if 'pixel_values' in inputs:
                        inputs['pixel_values'] = inputs['pixel_values'].half()
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                recon_embeddings = outputs.last_hidden_state
                
                # 计算距离
                distances = []
                for idx in range(len(recon_images)):
                    distance = self.compute_distance(original_embedding, recon_embeddings[idx:idx+1])[0]
                    distances.append(distance)
                
                avg_distance = np.mean(distances)
                features.append([[avg_distance], membership])
            
        return features
    
    def extract_multistage_features(self, dataset, sample_dir, num_samples=3, batch_size=8):
        """提取多阶段6维特征向量（创新方法）- 批量优化版本"""
        print("Extracting multistage 6D features...")
        
        # 如果使用离线模式
        if hasattr(self, 'use_offline') and self.use_offline:
            return self.offline_extractor.extract_multistage_features(dataset, sample_dir, num_samples)
        
        features = []
        
        # 批量处理原始图像
        original_images = [dataset["image"][i].convert("RGB") for i in range(len(dataset["image"]))]
        
        # 分批处理原始图像特征提取
        original_embeddings = []
        for batch_start in tqdm(range(0, len(original_images), batch_size), desc="Extracting original features"):
            batch_end = min(batch_start + batch_size, len(original_images))
            batch_images = original_images[batch_start:batch_end]
            
            # 批量处理
            inputs = self.feature_extractor(batch_images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # 转换为半精度以匹配模型
            if self.device.type == "cuda":
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].half()
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state
            
            for i in range(len(batch_images)):
                original_embeddings.append(batch_embeddings[i:i+1])
        
        # 处理多阶段重构图像
        for i in tqdm(range(len(dataset["image"])), desc="Processing multistage images"):
            original_embedding = original_embeddings[i]
            
            # 动态分配标签：前50%为成员(1)，后50%为非成员(0)
            membership = 1 if i < len(dataset["image"]) // 2 else 0
            
            # 提取三个阶段的特征
            stage_distances = []
            
            for stage_idx in range(1, 4):  # s1, s2, s3
                # 批量加载该阶段的图像
                stage_images = []
                for j in range(num_samples):
                    filename = f"image_{i+1:02}_s{stage_idx}_{j+1:02}.jpg"
                    image_path = os.path.join(sample_dir, filename)
                    
                    if os.path.exists(image_path):
                        recon_image = Image.open(image_path).convert("RGB")
                        stage_images.append(recon_image)
                
                if stage_images:
                    # 批量处理该阶段图像
                    inputs = self.feature_extractor(stage_images, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    
                    # 转换为半精度以匹配模型
                    if self.device.type == "cuda":
                        if 'pixel_values' in inputs:
                            inputs['pixel_values'] = inputs['pixel_values'].half()
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    stage_embeddings = outputs.last_hidden_state
                    
                    # 计算该阶段的平均距离
                    stage_dist = []
                    for idx in range(len(stage_images)):
                        distance = self.compute_distance(original_embedding, stage_embeddings[idx:idx+1])[0]
                        stage_dist.append(distance)
                    
                    avg_distance = np.mean(stage_dist)
                    stage_distances.append(avg_distance)
                else:
                    stage_distances.append(0.0)
            
            # 构建6维特征向量
            if len(stage_distances) >= 3:
                dist_50 = stage_distances[0]    # 50%时间步距离
                dist_80 = stage_distances[1]    # 80%时间步距离
                dist_100 = stage_distances[2]   # 100%时间步距离
                
                # 计算变化率
                rate1 = dist_80 - dist_50       # 50%→80%变化率
                rate2 = dist_100 - dist_80      # 80%→100%变化率
                
                # 计算改善比例
                improvement = (dist_50 - dist_100) / (dist_50 + 1e-8)
                
                feature_vector = [dist_50, dist_80, dist_100, rate1, rate2, improvement]
                features.append(feature_vector + [membership])
            else:
                print(f"Warning: Insufficient stage data for sample {i+1}")
        
        return features

def main():
    # 内置参数配置
    config = {
        "data_path": "data/test_dataset.pt",
        "single_image_dir": "result/single_stage_images",
        "multi_image_dir": "result/multi_stage_images",
        "single_features_path": "result/single_stage_features.pt",
        "multi_features_path": "result/multi_stage_features.pt",
        "image_encoder": "deit",  # deit, beit, vit
        "distance_method": "cosine",  # cosine, euclidean
        "num_samples": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print("=" * 60)
    print("多阶段重构攻击 - 特征提取模块")
    print("=" * 60)
    
    # 检查必要文件
    required_paths = [
        config["data_path"],
        config["single_image_dir"],
        config["multi_image_dir"]
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Error: Required path not found: {path}")
            return
    
    # 初始化特征提取器
    extractor = MultiStageFeatureExtractor(
        image_encoder=config["image_encoder"],
        method=config["distance_method"],
        device=config["device"]
    )
    
    try:
        extractor.load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 加载数据集
    print(f"Loading dataset from {config['data_path']}...")
    dataset = Dataset.from_dict(safe_torch_load(config["data_path"]))
    print(f"Loaded {len(dataset)} samples")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(config["single_features_path"]), exist_ok=True)
    
    # 提取单阶段特征
    print("\nStep 1: Extracting single-stage features (baseline)...")
    single_features = extractor.extract_single_stage_features(
        dataset=dataset,
        sample_dir=config["single_image_dir"],
        num_samples=config["num_samples"]
    )
    
    torch.save(single_features, config["single_features_path"])
    print(f"Single-stage features saved: {config['single_features_path']}")
    print(f"Feature dimension: 1, Samples: {len(single_features)}")
    
    # 提取多阶段特征
    print("\nStep 2: Extracting multistage features (our method)...")
    multi_features = extractor.extract_multistage_features(
        dataset=dataset,
        sample_dir=config["multi_image_dir"],
        num_samples=config["num_samples"]
    )
    
    torch.save(multi_features, config["multi_features_path"])
    print(f"Multistage features saved: {config['multi_features_path']}")
    print(f"Feature dimension: 6, Samples: {len(multi_features)}")
    
    # 显示特征示例
    if single_features:
        print(f"\nSingle-stage feature example: {single_features[0]}")
    if multi_features:
        print(f"Multi-stage feature example: {multi_features[0]}")
    
    print("\n" + "=" * 60)
    print("特征提取完成!")
    print("接下来请运行攻击训练: python train.py")
    print("=" * 60)

if __name__ == "__main__":
    main()