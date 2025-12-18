# Source Code Module Description

This directory contains all the core code files for the multi-stage reconstruction attack system.

### Pseudocode for this Experiment
Algorithm 1 Multi-stage Reconstruction-based Membership Inference Attack

Input:
  Private dataset Di for each client, 
  Initialized model ω, 
  Teacher model T,
  Number of clients N, 
  Global rounds T

Output:
  Robust global model.

for t = 1, 2, ..., T do
    for i = 1, ..., N in parallel do
        Send global model ω' to local client i
        ω' ← LocalUpdate(ω')
    end for
    L(ω) ← Σi∈[N] Di Σi∈[N] Di Li(ω')   (1)
end for

LocalUpdate(ω') 
for each local epoch do
    for each batch (xi, yi) of Di do
        /* Adversarial examples generation */
        xadv i ← xi + δ (2)
        /* Clean examples augmentation */
        xij ← λxi + (1 - λ)xj (3)
        /* Adversarial examples augmentation */
        xadv i ← xij + λ (1 - λ) xadv i (4)
        /* Vanilla mixture knowledge distillation */
        L VKD ← KL(ziij, zijadv) + KL(ziij, zisj) (5)
        /* Adversarial mixture knowledge distillation */
        L AKD ← KL(ziadv, zisadv) + KL(zijadv, zisadv) (6)
        /* Consistency regularization */
        L ALG ← λadv ||zsq - zgq||^2 (7)
        /* Overall local objective for each client */
        L ← α L VKD + (1 - α) L AKD + λ L ALG (8)
    end for
end for

return ωi

## 📁 File List

### 🎯 Core Attack Module
**inference.py** - Multi-stage image inference generation (enhanced to support three-stage reconstruction)
**cal_embedding.py** - Feature embedding computation (enhanced to support 6-dimensional feature vectors)
**test_accuracy.py** - Attack accuracy testing (enhanced to support multiple classifiers)

### 🛠️ Auxiliary Training Modules
**train_text_to_image_lora.py** - LoRA text-to-image training
**blip_finetune.py** - BLIP model fine-tuning
**build_caption.py** - Image caption construction

### 🔧 Utility Modules
**download_coco.py** - COCO dataset download tool
**kandinsky2_2_inference.py** - Kandinsky model inference

### 📊 Simplified Interface Modules
**extract_features.py** - Simplified interface for feature extraction (calls cal_embedding.py)
**classifier.py** - Simplified interface for classifier training (calls test_accuracy.py)

## 🚀 How to Use

### Complete Original Workflow
```bash
# 1. Download Dataset
python download_coco.py
# 2. Train LoRA Model
python train_text_to_image_lora.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ...
# 3. Generate Images (Multi-stage)
python inference.py --multistage --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ...
# 4. Extract Features (6-dimensional)
python cal_embedding.py --multistage --data_dir=... --sample_file=...
# 5. Test Attacks
python test_accuracy.py --method=random_forest --multistage ...
```

### Simplified Workflow
```bash
# Using the Simplified Interface
python extract_features.py # Automatically calls cal_embedding.py
python classifier.py # Automatically calls test_accuracy.py
```

## 🔄 File Relationships
``` Original Core Files:
├── inference.py (Image Generation)
├── cal_embedding.py (Feature Extraction)
├── test_accuracy.py (Classification Test)
└── Auxiliary Files...

Simplified Interface Files:
├── extract_features.py → Calls cal_embedding.py
├── classifier.py → Calls test_accuracy.py
└── Provides a more user-friendly API
```
## ⚙️ Parameter Explanation

### New Parameters in inference.py
`--multistage` - Enables three-stage reconstruction (50%, 80%, 100%)
`--seed` - Random seed

### New Parameters in cal_embedding.py
`--multistage` - Extracts 6-dimensional feature vectors instead of 1-dimensional ones

### New Parameters in test_accuracy.py
`--multistage` - Processes 6-dimensional feature data
`--method` - Supports more classifiers (random_forest, svm, logistic, etc.)

## 🧪 Testing Suggestions
1. **Functional Testing**: First run the simplified interface to ensure basic functionality.
2. **Complete Testing**: Then perform a complete process test using the original files.
3. **Performance Comparison**: Compare the performance differences between single-stage and multi-stage tests.

## 📝 Notes
Maintain complete functionality in the original files for backward compatibility.
Simplified interface for a better user experience
All files support multi-stage and single-stage modes
Configuration parameters can be managed centrally in `../config/config.py`