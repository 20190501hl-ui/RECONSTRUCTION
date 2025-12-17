from transformers import DeiTFeatureExtractor, DeiTModel, DeformableDetrModel, AutoImageProcessor, BeitModel, EfficientFormerModel, ViTModel
import torch
import pickle
from datasets import Dataset
from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np

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

def parse_args():
    parser = argparse.ArgumentParser(description="get image embedding")
    parser.add_argument("--data_dir",type=str,default=None)
    parser.add_argument("--sample_file",type=str,default=None)
    parser.add_argument("--membership",type=int,default=None)
    parser.add_argument("--img_num",type=int,default=None)
    parser.add_argument("--gpu",type=int,default=None)
    parser.add_argument("--save_dir",type=str,default=None)
    parser.add_argument("--method",type=str,default="cosine")
    parser.add_argument("--image_encoder",type=str,default="deit")
    parser.add_argument("--similarity_score_dim",type=int,default=1)
    parser.add_argument(
        "--multistage",
        action="store_true",
        help="Enable multistage feature extraction (6-dim features)"
    )
    args = parser.parse_args()

    return args

def compute_scores(emb_one, emb_two, method, similarity_score_dim):
    """Computes distance/similarity between two vectors."""
    if method == "cosine":
        scores = torch.nn.functional.cosine_similarity(emb_one, emb_two, dim=similarity_score_dim)
    elif method == "euclidean":
        scores = torch.nn.functional.pairwise_distance(emb_one, emb_two, dim=similarity_score_dim)
    elif method == "manhattan":
        scores = torch.sum(torch.abs(emb_one - emb_two), dim=similarity_score_dim)
    elif method == "hamming":
        emb_one = emb_one.int()
        emb_two = emb_two.int()
        scores = torch.sum((emb_one ^ emb_two), dim=similarity_score_dim)
    return scores.cpu().detach().numpy()

def extract_multistage_features(original_embedding, sample_dir, image_idx, num_samples, 
                               feature_extractor, model, device, method, similarity_score_dim):
    """
    Extract 6-dimensional features from three reconstruction stages.
    
    Returns:
        List of 6 features: [dist_50%, dist_80%, dist_100%, rate1, rate2, improvement_ratio]
    """
    stage_distances = []
    
    # Extract features for each stage (s1=50%, s2=80%, s3=100%)
    for stage_idx in range(1, 4):
        stage_scores = []
        
        for j in range(num_samples):
            # Format: image_01_s1_01.jpg
            filename = f"image_{image_idx+1:02}_s{stage_idx}_{j+1:02}.jpg"
            save_path = os.path.join(sample_dir, filename)
            
            if not os.path.exists(save_path):
                print(f"Warning: File {filename} not found, skipping...")
                continue
                
            img = Image.open(save_path).convert("RGB")
            inputs = feature_extractor(img, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            stage_embedding = outputs.last_hidden_state
            
            # Compute distance between original and reconstructed
            distance = compute_scores(original_embedding, stage_embedding, method, similarity_score_dim)[0]
            stage_scores.append(distance)
        
        if stage_scores:  # Only add if we have valid scores
            avg_distance = np.mean(stage_scores)
            stage_distances.append(avg_distance)
        else:
            print(f"Warning: No valid samples found for stage {stage_idx}")
            stage_distances.append(0.0)  # Default value
    
    # Build 6-dimensional feature vector
    if len(stage_distances) >= 3:
        dist_50 = stage_distances[0]    # Distance at 50% timestep
        dist_80 = stage_distances[1]    # Distance at 80% timestep  
        dist_100 = stage_distances[2]   # Distance at 100% timestep
        
        # Change rates between stages
        rate1 = dist_80 - dist_50       # Change rate from 50% to 80%
        rate2 = dist_100 - dist_80      # Change rate from 80% to 100%
        
        # Improvement ratio (how much the distance improved from start to end)
        improvement_ratio = (dist_50 - dist_100) / (dist_50 + 1e-8) if dist_50 > 0 else 0.0
        
        features = [dist_50, dist_80, dist_100, rate1, rate2, improvement_ratio]
    else:
        print("Warning: Not enough stage distances, using default features")
        features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return features

def main():
    args = parse_args()
    
    dataset = Dataset.from_dict(safe_torch_load(args.data_dir))
    
    # Load feature extractor and model
    if args.image_encoder=="deit":
        feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-384")
        model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-384", add_pooling_layer=False)
    elif args.image_encoder=="detr":
        feature_extractor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        model = DeformableDetrModel.from_pretrained("SenseTime/deformable-detr") 
    elif args.image_encoder=="beit":
        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k") 
    elif args.image_encoder=="eformer":
        feature_extractor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
        model = EfficientFormerModel.from_pretrained("snap-research/efficientformer-l1-300")  
    elif args.image_encoder=="vit":
        feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Using {args.image_encoder} encoder on device: {device}")
    print(f"Multistage mode: {args.multistage}")
    
    scores = []
    
    for i in tqdm(range(len(dataset["image"])), desc="Processing images"):
        # Extract original image embedding
        image_target = dataset["image"][i].convert("RGB")
        inputs = feature_extractor(image_target, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        original_embedding = outputs.last_hidden_state
        
        if args.multistage:
            # Extract 6-dimensional multistage features
            features = extract_multistage_features(
                original_embedding, args.sample_file, i, args.img_num,
                feature_extractor, model, device, args.method, args.similarity_score_dim
            )
            features.append(args.membership)
            scores.append(features)
            
        else:
            # Original single-stage approach
            temp = []
            for j in range(args.img_num):
                filename = f"image_{i+1:02}_{j+1:02}.jpg"
                save_path = os.path.join(args.sample_file, filename)
                
                if not os.path.exists(save_path):
                    print(f"Warning: File {filename} not found, skipping...")
                    continue
                    
                img = Image.open(save_path).convert("RGB")
                inputs = feature_extractor(img, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                reconstructed_embedding = outputs.last_hidden_state
                
                distance = compute_scores(original_embedding, reconstructed_embedding, 
                                        args.method, args.similarity_score_dim)[0]
                temp.append(distance)
            
            if temp:  # Only process if we have valid samples
                temp = np.array(temp)
                average = np.mean(temp, axis=0)
                result = [average.tolist()]
                result.append(args.membership)
                scores.append(result)
            else:
                print(f"Warning: No valid samples found for image {i+1}")
    
    # Save results
    print(f"Processed {len(scores)} samples")
    print(f"Saving results to: {args.save_dir}")
    torch.save(scores, args.save_dir)
    
    # Print some statistics
    if scores:
        if args.multistage:
            print(f"Sample multistage features shape: {len(scores[0])-1} dimensions")
            print(f"First sample features: {scores[0][:-1]}")
        else:
            print(f"Sample single-stage features shape: {len(scores[0][0])} dimensions")
            print(f"First sample features: {scores[0][0]}")
    
    print("Feature extraction completed!")

if __name__ == "__main__":
    args = parse_args()
    main()



