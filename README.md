# 源代码模块说明

本目录包含多阶段重构攻击系统的所有核心代码文件。

## 📁 文件列表

### 🎯 核心攻击模块
- **inference.py** - 多阶段图像推理生成（已增强支持三阶段重构）
- **cal_embedding.py** - 特征嵌入计算（已增强支持6维特征向量）  
- **test_accuracy.py** - 攻击准确率测试（已增强支持多种分类器）

### 🛠️ 辅助训练模块
- **train_text_to_image_lora.py** - LoRA文本到图像训练
- **blip_finetune.py** - BLIP模型微调
- **build_caption.py** - 图像标题构建

### 🔧 工具模块
- **download_coco.py** - COCO数据集下载工具
- **kandinsky2_2_inference.py** - Kandinsky模型推理

### 📊 简化接口模块  
- **extract_features.py** - 特征提取简化接口（调用cal_embedding.py）
- **classifier.py** - 分类器训练简化接口（调用test_accuracy.py）

## 🚀 使用方法

### 完整原始流程
```bash
# 1. 下载数据集
python download_coco.py

# 2. 训练LoRA模型  
python train_text_to_image_lora.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ...

# 3. 生成图像（多阶段）
python inference.py --multistage --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ...

# 4. 提取特征（6维）
python cal_embedding.py --multistage --data_dir=... --sample_file=...

# 5. 测试攻击
python test_accuracy.py --method=random_forest --multistage ...
```

### 简化流程
```bash
# 使用简化接口
python extract_features.py  # 自动调用cal_embedding.py
python classifier.py        # 自动调用test_accuracy.py
```

## 🔄 文件关系

```
原始核心文件:
├── inference.py (图像生成) 
├── cal_embedding.py (特征提取)
├── test_accuracy.py (分类测试)
└── 辅助文件...

简化接口文件:
├── extract_features.py → 调用 cal_embedding.py
├── classifier.py → 调用 test_accuracy.py
└── 提供更友好的API
```

## ⚙️ 参数说明

### inference.py 新增参数
- `--multistage` - 启用三阶段重构（50%, 80%, 100%）
- `--seed` - 随机种子

### cal_embedding.py 新增参数
- `--multistage` - 提取6维特征向量而非1维

### test_accuracy.py 新增参数
- `--multistage` - 处理6维特征数据
- `--method` - 支持更多分类器（random_forest, svm, logistic等）

## 🧪 测试建议

1. **功能测试**: 先运行简化接口确保基本功能正常
2. **完整测试**: 再用原始文件进行完整流程测试
3. **性能对比**: 对比单阶段vs多阶段的性能差异

## 📝 注意事项

- 原始文件保持完整功能，向后兼容
- 简化接口提供更好的用户体验
- 所有文件都支持多阶段和单阶段模式
- 配置参数可在 `../config/config.py` 中统一管理