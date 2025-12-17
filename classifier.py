#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多阶段攻击分类器模块
训练和评估成员推理攻击分类器
"""

import os
import pickle
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib

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

class MembershipInferenceAttacker:
    def __init__(self, method="random_forest"):
        self.method = method
        self.classifier = None
        self.scaler = StandardScaler()
        
    def _init_classifier(self):
        """初始化分类器"""
        if self.method == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        elif self.method == "svm":
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        elif self.method == "logistic":
            self.classifier = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def prepare_data(self, features_path, is_multistage=False):
        """准备训练数据"""
        print(f"Loading features from {features_path}...")
        data = safe_torch_load(features_path)
        
        if is_multistage:
            # 多阶段：[feat1, feat2, feat3, feat4, feat5, feat6, membership]
            features = np.array([item[:-1] for item in data])
            labels = np.array([item[-1] for item in data])
        else:
            # 单阶段：[[distance], membership]
            features = np.array([item[0] for item in data])
            labels = np.array([item[-1] for item in data])
        
        print(f"Features shape: {features.shape}")
        print(f"Labels distribution - Members: {np.sum(labels)}, Non-members: {len(labels) - np.sum(labels)}")
        
        return features, labels
    
    def train(self, train_features, train_labels):
        """训练分类器"""
        print(f"Training {self.method} classifier...")
        
        # 标准化特征
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # 初始化并训练分类器
        self._init_classifier()
        self.classifier.fit(train_features_scaled, train_labels)
        
        print("Training completed!")
    
    def evaluate(self, test_features, test_labels):
        """评估分类器性能"""
        print("Evaluating classifier...")
        
        # 标准化测试特征
        test_features_scaled = self.scaler.transform(test_features)
        
        # 检查测试集中是否有足够的类别
        unique_labels = np.unique(test_labels)
        if len(unique_labels) < 2:
            print(f"Warning: Test set only contains {len(unique_labels)} class(es), cannot evaluate properly")
            return {
                'accuracy': 0.5,
                'auc_roc': 0.5,
                'tpr_1percent_fpr': 0.0,
                'fpr': np.array([0, 1]),
                'tpr': np.array([0, 1])
            }
        
        # 预测
        y_pred = self.classifier.predict(test_features_scaled)
        y_proba = self.classifier.predict_proba(test_features_scaled)
        
        # 检查是否有两个类别的概率
        if y_proba.shape[1] == 1:
            # 只有一个类别，补充第二个类别概率
            y_proba_full = np.zeros((len(y_proba), 2))
            if unique_labels[0] == 0:
                y_proba_full[:, 0] = y_proba[:, 0]
                y_proba_full[:, 1] = 1 - y_proba[:, 0]
            else:
                y_proba_full[:, 1] = y_proba[:, 0]
                y_proba_full[:, 0] = 1 - y_proba[:, 0]
            y_proba_positive = y_proba_full[:, 1]
        else:
            y_proba_positive = y_proba[:, 1]
        
        # 计算指标
        accuracy = accuracy_score(test_labels, y_pred)
        roc_auc = roc_auc_score(test_labels, y_proba_positive)
        
        # 计算TPR@1%FPR
        fpr, tpr, _ = roc_curve(test_labels, y_proba_positive)
        idx_1percent = np.argmin(np.abs(fpr - 0.01))
        tpr_1percent = tpr[idx_1percent]
        
        results = {
            'accuracy': accuracy,
            'auc_roc': roc_auc,
            'tpr_1percent_fpr': tpr_1percent,
            'fpr': fpr,
            'tpr': tpr
        }
        
        return results
    
    def get_feature_importance(self):
        """获取特征重要性（仅随机森林）"""
        if self.method == "random_forest" and hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        return None
    
    def save_model(self, save_path):
        """保存训练好的模型"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'method': self.method
        }
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")

def compare_methods(single_features_path, multi_features_path):
    """对比单阶段和多阶段方法的性能"""
    print("\n" + "=" * 60)
    print("性能对比分析")
    print("=" * 60)
    
    results = {}
    
    # 测试单阶段方法
    print("\n--- 单阶段方法评估 ---")
    single_attacker = MembershipInferenceAttacker("random_forest")
    single_features, single_labels = single_attacker.prepare_data(single_features_path, False)
    
    # 使用分层抽样确保训练测试集中都有两个类别
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(
        single_features, single_labels, 
        test_size=0.5, 
        random_state=42, 
        stratify=single_labels
    )
    
    single_attacker.train(train_X, train_y)
    single_results = single_attacker.evaluate(test_X, test_y)
    results['single'] = single_results
    
    print(f"单阶段结果:")
    print(f"  准确率: {single_results['accuracy']:.4f}")
    print(f"  AUC-ROC: {single_results['auc_roc']:.4f}")
    print(f"  TPR@1%FPR: {single_results['tpr_1percent_fpr']:.4f}")
    
    # 测试多阶段方法
    print("\n--- 多阶段方法评估 ---")
    multi_attacker = MembershipInferenceAttacker("random_forest")
    multi_features, multi_labels = multi_attacker.prepare_data(multi_features_path, True)
    
    # 使用相同的分层分割方法
    train_X, test_X, train_y, test_y = train_test_split(
        multi_features, multi_labels, 
        test_size=0.5, 
        random_state=42, 
        stratify=multi_labels
    )
    
    multi_attacker.train(train_X, train_y)
    multi_results = multi_attacker.evaluate(test_X, test_y)
    results['multi'] = multi_results
    
    print(f"多阶段结果:")
    print(f"  准确率: {multi_results['accuracy']:.4f}")
    print(f"  AUC-ROC: {multi_results['auc_roc']:.4f}")
    print(f"  TPR@1%FPR: {multi_results['tpr_1percent_fpr']:.4f}")
    
    # 显示改进幅度
    acc_improvement = multi_results['accuracy'] - single_results['accuracy']
    auc_improvement = multi_results['auc_roc'] - single_results['auc_roc']
    
    print(f"\n--- 性能改进 ---")
    print(f"准确率提升: +{acc_improvement:.4f} ({acc_improvement/single_results['accuracy']*100:.1f}%)")
    print(f"AUC提升: +{auc_improvement:.4f} ({auc_improvement/single_results['auc_roc']*100:.1f}%)")
    
    # 显示特征重要性
    feature_importance = multi_attacker.get_feature_importance()
    if feature_importance is not None:
        feature_names = ['Dist@50%', 'Dist@80%', 'Dist@100%', 'Rate1', 'Rate2', 'Improvement']
        print(f"\n--- 特征重要性 ---")
        for name, importance in zip(feature_names, feature_importance):
            print(f"  {name}: {importance:.4f}")
    
    return results

def main():
    # 内置参数配置
    config = {
        "single_features_path": "result/single_stage_features.pt",
        "multi_features_path": "result/multi_stage_features.pt",
        "model_save_dir": "result/models",
        "method": "random_forest"  # random_forest, svm, logistic
    }
    
    print("=" * 60)
    print("多阶段重构攻击 - 分类器训练模块")
    print("=" * 60)
    
    # 检查特征文件
    required_files = [config["single_features_path"], config["multi_features_path"]]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Feature file not found: {file_path}")
            print("Please run extract_features.py first.")
            return
    
    # 创建模型保存目录
    os.makedirs(config["model_save_dir"], exist_ok=True)
    
    # 对比分析
    try:
        results = compare_methods(
            config["single_features_path"],
            config["multi_features_path"]
        )
        
        # 保存结果
        torch.save(results, os.path.join(config["model_save_dir"], "comparison_results.pt"))
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"结果已保存到: {config['model_save_dir']}")
        print("接下来请运行可视化: python visualize.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {e}")
        return

if __name__ == "__main__":
    main()