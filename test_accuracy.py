from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
import torch
import pickle
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score, roc_curve
import argparse
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
    parser = argparse.ArgumentParser(description="get attack accuracy")
    parser.add_argument("--target_member_dir",type=str,default=None)
    parser.add_argument("--target_non_member_dir",type=str,default=None)
    parser.add_argument("--shadow_member_dir",type=str,default=None)
    parser.add_argument("--shadow_non_member_dir",type=str,default=None)
    parser.add_argument("--method",type=str,default="classifier",
                       help="classifier, random_forest, svm, logistic, distribution, threshold")
    parser.add_argument(
        "--multistage", 
        action="store_true",
        help="Use multistage 6-dim features instead of single-stage features"
    )
    parser.add_argument("--save_plots", action="store_true", help="Save visualization plots")
    args = parser.parse_args()
    return args

def process_data(args):
    """Process data for both single-stage and multi-stage features"""
    t_m = safe_torch_load(args.target_member_dir)
    t_n_m = safe_torch_load(args.target_non_member_dir)
    s_m = safe_torch_load(args.shadow_member_dir)
    s_n_m = safe_torch_load(args.shadow_non_member_dir)

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    
    print(f"Loaded data shapes:")
    print(f"  Target members: {len(t_m)}")
    print(f"  Target non-members: {len(t_n_m)}")  
    print(f"  Shadow members: {len(s_m)}")
    print(f"  Shadow non-members: {len(s_n_m)}")

    # Process shadow data for training
    for dataset, label in [(s_m, 1), (s_n_m, 0)]:
        for item in dataset:
            if args.multistage:
                # For multistage: item = [feat1, feat2, feat3, feat4, feat5, feat6, membership]
                features = item[:-1]  # First 6 dimensions
                assert len(features) == 6, f"Expected 6 features, got {len(features)}"
            else:
                # For single-stage: item = [[features_list], membership]  
                features = item[0] if isinstance(item[0], list) else [item[0]]
                
            train_features.append(features)
            train_labels.append(label)

    # Process target data for testing
    for dataset, label in [(t_m, 1), (t_n_m, 0)]:
        for item in dataset:
            if args.multistage:
                # For multistage: item = [feat1, feat2, feat3, feat4, feat5, feat6, membership]
                features = item[:-1]  # First 6 dimensions
                assert len(features) == 6, f"Expected 6 features, got {len(features)}"
            else:
                # For single-stage: item = [[features_list], membership]
                features = item[0] if isinstance(item[0], list) else [item[0]]
                
            test_features.append(features)
            test_labels.append(label)

    train_features = np.array(train_features)
    test_features = np.array(test_features)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    print(f"Processed feature shapes:")
    print(f"  Train features: {train_features.shape}")
    print(f"  Test features: {test_features.shape}")
    print(f"  Train labels: {train_labels.shape} (members: {np.sum(train_labels)}, non-members: {len(train_labels) - np.sum(train_labels)})")
    print(f"  Test labels: {test_labels.shape} (members: {np.sum(test_labels)}, non-members: {len(test_labels) - np.sum(test_labels)})")

    # Feature scaling
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    
    return train_features, train_labels, test_features, test_labels


class DefineClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(DefineClassifier, self).__init__()
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )


        self.layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Layer 2
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Layer 3
        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Layer 4
        self.layer5 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )

        self.layer6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        self.out_layer = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return self.out_layer(x)






def evaluate_classifier(y_true, y_pred, y_proba=None, method_name="Classifier"):
    """Evaluate classifier performance with comprehensive metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    
    results = {
        'method': method_name,
        'accuracy': accuracy,
        'auc_roc': 0.0,
        'tpr_1percent_fpr': 0.0,
        'tpr_01percent_fpr': 0.0
    }
    
    if y_proba is not None:
        # ROC curve analysis
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        results['auc_roc'] = roc_auc
        
        # TPR at specific FPR levels
        desired_fprs = [0.01, 0.001]  # 1% and 0.1%
        for desired_fpr in desired_fprs:
            idx = np.argmin(np.abs(fpr - desired_fpr))
            tpr_at_fpr = tpr[idx]
            if desired_fpr == 0.01:
                results['tpr_1percent_fpr'] = tpr_at_fpr
            else:
                results['tpr_01percent_fpr'] = tpr_at_fpr
    
    return results

def run_random_forest(train_features, train_labels, test_features, test_labels):
    """Run Random Forest classifier"""
    print("\n=== Random Forest Classifier ===")
    
    clf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    clf.fit(train_features, train_labels)
    
    # Predictions
    y_pred = clf.predict(test_features)
    y_proba = clf.predict_proba(test_features)[:, 1]
    
    # Feature importance analysis
    if hasattr(clf, 'feature_importances_'):
        importance = clf.feature_importances_
        feature_names = ['Dist@50%', 'Dist@80%', 'Dist@100%', 'Rate1', 'Rate2', 'Improvement']
        
        print("Feature Importance:")
        for name, imp in zip(feature_names[:len(importance)], importance):
            print(f"  {name}: {imp:.4f}")
    
    return evaluate_classifier(test_labels, y_pred, y_proba, "Random Forest")

def run_svm(train_features, train_labels, test_features, test_labels):
    """Run SVM classifier"""
    print("\n=== SVM Classifier ===")
    
    clf = SVC(
        kernel='rbf',
        probability=True,  # Enable probability estimates
        random_state=42
    )
    clf.fit(train_features, train_labels)
    
    y_pred = clf.predict(test_features)
    y_proba = clf.predict_proba(test_features)[:, 1]
    
    return evaluate_classifier(test_labels, y_pred, y_proba, "SVM")

def run_logistic_regression(train_features, train_labels, test_features, test_labels):
    """Run Logistic Regression"""
    print("\n=== Logistic Regression ===")
    
    clf = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    clf.fit(train_features, train_labels)
    
    y_pred = clf.predict(test_features)
    y_proba = clf.predict_proba(test_features)[:, 1]
    
    return evaluate_classifier(test_labels, y_pred, y_proba, "Logistic Regression")

def main(train_features, train_labels, test_features, test_labels, args):
    """Main function to run different classifiers"""
    
    print(f"\n{'='*50}")
    print(f"MEMBERSHIP INFERENCE ATTACK EVALUATION")
    print(f"{'='*50}")
    print(f"Feature mode: {'Multistage (6-dim)' if args.multistage else 'Single-stage'}")
    print(f"Method: {args.method}")
    
    results = []
    
    if args.method == "random_forest":
        result = run_random_forest(train_features, train_labels, test_features, test_labels)
        results.append(result)
        
    elif args.method == "svm":
        result = run_svm(train_features, train_labels, test_features, test_labels)
        results.append(result)
        
    elif args.method == "logistic":
        result = run_logistic_regression(train_features, train_labels, test_features, test_labels)
        results.append(result)
        
    elif args.method == "classifier":
        # Original neural network classifier
        train_labels_tensor = torch.tensor(train_labels) 
        train_features_tensor = torch.tensor(train_features)
        test_labels_tensor = torch.tensor(test_labels) 
        test_features_tensor = torch.tensor(test_features)
        
        input_dim = train_features_tensor.shape[1]
        train_features_tensor = train_features_tensor.to(torch.float32)
        test_features_tensor = test_features_tensor.to(torch.float32)
        
        model = DefineClassifier(input_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  

        num_epochs = 200  # Reduced for faster training
        best_auc = 0
        best_accuracy = 0
        
        print("\n=== Neural Network Classifier ===")
        print("Training...")
        
        for epoch in range(num_epochs):
            model.train()  
            outputs = model(train_features_tensor)
            loss = criterion(outputs, train_labels_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(test_features_tensor)
                    _, test_predicted = torch.max(test_outputs.data, 1)
                    test_correct = (test_predicted == test_labels_tensor).sum().item()
                    test_accuracy = test_correct / test_labels_tensor.size(0)
                    
                    # ROC analysis
                    test_proba = torch.softmax(test_outputs, dim=1)[:, 1].numpy()
                    fpr, tpr, _ = roc_curve(test_labels, test_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    if roc_auc > best_auc:
                        best_auc = roc_auc
                        best_accuracy = test_accuracy
                
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss.item():.4f}, "
                      f"Test Acc={test_accuracy:.4f}, AUC={roc_auc:.4f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_features_tensor)
            test_proba = torch.softmax(test_outputs, dim=1)[:, 1].numpy()
            test_pred = torch.argmax(test_outputs, dim=1).numpy()
        
        result = evaluate_classifier(test_labels, test_pred, test_proba, "Neural Network")
        results.append(result)
        
    elif args.method == "distribution":
        # Distribution-based method (existing code)
        print("\n=== Distribution-based Method ===")
        class0_samples = train_features[train_labels == 0]
        mean_0 = np.mean(class0_samples, axis=0)
        cov_0 = np.cov(class0_samples, rowvar=False)
        
        class1_samples = train_features[train_labels == 1]
        mean_1 = np.mean(class1_samples, axis=0)
        cov_1 = np.cov(class1_samples, rowvar=False)
        
        # Add regularization
        reg_value = 1e-5
        cov_0 += reg_value * np.eye(cov_0.shape[0])
        cov_1 += reg_value * np.eye(cov_1.shape[0])
        
        rv_0 = multivariate_normal(mean_0, cov_0)
        rv_1 = multivariate_normal(mean_1, cov_1)
        
        # Predictions
        scores = [p_1 - p_0 for p_0, p_1 in zip(rv_0.logpdf(test_features), rv_1.logpdf(test_features))]
        predictions = [1 if s > 0 else 0 for s in scores]
        
        result = evaluate_classifier(test_labels, predictions, scores, "Distribution-based")
        results.append(result)
        
    elif args.method == "threshold":
        # Threshold-based method (existing code)
        print("\n=== Threshold-based Method ===")
        data_means = np.max(train_features, axis=1)
        sorted_means = np.sort(data_means)
        potential_thresholds = (sorted_means[:-1] + sorted_means[1:]) / 2
        
        accuracies = []
        for threshold in potential_thresholds:
            predicted_labels = np.where(data_means > threshold, 1, 0)
            accuracy = np.mean(predicted_labels == train_labels)
            accuracies.append(accuracy)

        best_threshold = potential_thresholds[np.argmax(accuracies)]
        
        # Test predictions
        data_means_test = np.max(test_features, axis=1)
        predictions = np.where(data_means_test > best_threshold, 1, 0)
        
        result = evaluate_classifier(test_labels, predictions, data_means_test, "Threshold-based")
        results.append(result)

    # Print summary results
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    
    for result in results:
        print(f"\nMethod: {result['method']}")
        print(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"  AUC-ROC: {result['auc_roc']:.4f}")
        print(f"  TPR@1%FPR: {result['tpr_1percent_fpr']:.4f}")
        print(f"  TPR@0.1%FPR: {result['tpr_01percent_fpr']:.4f}")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    train_features, train_labels, test_features, test_labels = process_data(args)
    results = main(train_features, train_labels, test_features, test_labels, args)
