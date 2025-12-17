# 多阶段重构攻击配置文件
# Multi-stage Reconstruction Attack Configuration

# 基础路径配置
DATA_CONFIG = {
    "dataset_path": "data/test_dataset.pt",        # 测试数据集路径
    "lora_model_path": "output_lora",              # LoRA模型路径
    "pretrained_model": "runwayml/stable-diffusion-v1-5"  # 预训练模型
}

# 图像生成配置
INFERENCE_CONFIG = {
    "num_images_per_sample": 3,      # 每个样本生成图像数量
    "inference_steps": 30,           # 推理步数
    "guidance_scale": 7.5,           # 引导比例
    "timestep_ratios": [0.5, 0.8, 1.0],  # 三阶段时间步比例 (50%, 80%, 100%)
    "seed": 1337,                    # 随机种子
    "device": "cuda"                 # 设备 ("cuda" 或 "cpu")
}

# 特征提取配置
FEATURE_CONFIG = {
    "image_encoder": "deit",         # 图像编码器 ("deit", "beit", "vit")
    "distance_method": "cosine",     # 距离计算方法 ("cosine", "euclidean")
    "feature_names": [               # 6维特征名称
        "Dist@50%",      # 50%时间步距离
        "Dist@80%",      # 80%时间步距离
        "Dist@100%",     # 100%时间步距离
        "Rate1",         # 50%→80%变化率
        "Rate2",         # 80%→100%变化率
        "Improvement"    # 总体改善比例
    ]
}

# 分类器配置
CLASSIFIER_CONFIG = {
    "methods": ["random_forest", "svm", "logistic"],  # 支持的分类器方法
    "default_method": "random_forest",                # 默认方法
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    },
    "svm": {
        "kernel": "rbf",
        "probability": True,
        "random_state": 42
    },
    "logistic": {
        "max_iter": 1000,
        "random_state": 42
    }
}

# 输出路径配置
OUTPUT_CONFIG = {
    "result_dir": "result",                           # 结果根目录
    "single_stage_images": "result/single_stage_images",    # 单阶段图像目录
    "multi_stage_images": "result/multi_stage_images",      # 多阶段图像目录
    "single_stage_features": "result/single_stage_features.pt",  # 单阶段特征文件
    "multi_stage_features": "result/multi_stage_features.pt",    # 多阶段特征文件
    "models_dir": "result/models",                    # 模型保存目录
    "visualization_dir": "result/visualizations"      # 可视化结果目录
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "figure_size": {
        "comparison": (12, 5),       # 性能对比图尺寸
        "trajectories": (15, 10),    # 轨迹图尺寸
        "distributions": (15, 10),   # 分布图尺寸
        "importance": (10, 6)        # 重要性图尺寸
    },
    "colors": {
        "member": "#2E86C1",         # 成员数据颜色（蓝色）
        "non_member": "#E74C3C",     # 非成员数据颜色（红色）
        "single_method": "#ff7f7f",  # 单阶段方法颜色
        "multi_method": "#7f7fff"    # 多阶段方法颜色
    },
    "dpi": 300,                      # 图片分辨率
    "format": "png"                  # 保存格式
}

# 实验配置
EXPERIMENT_CONFIG = {
    "train_test_split": 0.5,         # 训练测试分割比例
    "cross_validation_folds": 5,     # 交叉验证折数
    "performance_metrics": [         # 评估指标
        "accuracy",
        "auc_roc", 
        "tpr_at_1_percent_fpr"
    ]
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",                 # 日志级别
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "file": "result/experiment.log"  # 日志文件
}