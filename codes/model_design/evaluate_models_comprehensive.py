"""
完整的模型评估脚本 - 在model_design目录中，避免导入问题
包含CSI、POD、FAR、SSIM、MSE、MAE等所有指标
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

# 获取当前目录（model_design）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # codes目录

print("=" * 70)
print("雷达回波预测模型完整评估系统")
print("=" * 70)
print(f"当前目录: {CURRENT_DIR}")
print(f"项目根目录: {PROJECT_ROOT}")

# ================= 1. 导入同一目录下的模块 =================
# 由于在同一目录，可以直接导入
from lstm_cnn_model import LSTMCNN, SimpleViT
from train_lstm_cnn_model import RadarLazyDataset, CONFIG as TRAIN_CONFIG

print("✓ 成功导入所有模块")

# ================= 2. 配置参数 =================
EVAL_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_samples': 100,
    'batch_size': 4,
    'img_size': TRAIN_CONFIG.get('img_size', (256, 256)),
    'seq_len': TRAIN_CONFIG.get('seq_len', 20),
    'input_len': TRAIN_CONFIG.get('input_len', 10),
    'target_len': TRAIN_CONFIG.get('target_len', 10),
    
    # dBZ阈值
    'dbz_thresholds': [10, 15, 20, 25, 30, 35, 40],
    
    # 使用绝对路径
    'images_dir': os.path.join(PROJECT_ROOT, 'processed_data', 'images'),
    'list_file': os.path.join(PROJECT_ROOT, 'processed_data', 'train_list.txt'),
    'output_dir': os.path.join(PROJECT_ROOT, 'evaluation_results'),
    
    # 模型路径
    'model_paths': {
        'LSTM-CNN': os.path.join(PROJECT_ROOT, 'checkpoints', 'best_lstm_cnn_model.pth'),
        'ViT': os.path.join(PROJECT_ROOT, 'checkpoints', 'best_vit_model.pth'),
    }
}

# 创建输出目录
os.makedirs(EVAL_CONFIG['output_dir'], exist_ok=True)

print(f"\n配置:")
print(f"  设备: {EVAL_CONFIG['device']}")
print(f"  最大样本数: {EVAL_CONFIG['max_samples']}")
print(f"  Batch大小: {EVAL_CONFIG['batch_size']}")
print(f"  图像尺寸: {EVAL_CONFIG['img_size']}")
print(f"  dBZ阈值: {EVAL_CONFIG['dbz_thresholds']}")

# ================= 3. 评估指标计算类 =================
class ComprehensiveMetrics:
    """综合评估指标计算"""
    
    def __init__(self, thresholds=None, max_dbz=70.0):
        self.thresholds = thresholds or [15, 20, 25, 30, 35]
        self.max_dbz = max_dbz
    
    @staticmethod
    def normalize_to_dbz(image, max_dbz=70.0):
        """将归一化图像转换为dBZ值"""
        return image * max_dbz
    
    @staticmethod
    def binarize_image(image, threshold_dbz):
        """二值化图像"""
        return (image >= threshold_dbz).astype(np.float32)
    
    def calculate_classification_metrics(self, y_true, y_pred, threshold):
        """计算分类指标：CSI, POD, FAR"""
        # 转换为dBZ
        y_true_dbz = self.normalize_to_dbz(y_true)
        y_pred_dbz = self.normalize_to_dbz(y_pred)
        
        # 二值化
        y_true_bin = self.binarize_image(y_true_dbz, threshold)
        y_pred_bin = self.binarize_image(y_pred_dbz, threshold)
        
        # 展平
        y_true_flat = y_true_bin.flatten()
        y_pred_flat = y_pred_bin.flatten()
        
        # 计算混淆矩阵
        tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
        tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
        
        metrics = {}
        
        # CSI (Critical Success Index)
        if (tp + fp + fn) > 0:
            metrics['CSI'] = tp / (tp + fp + fn)
        else:
            metrics['CSI'] = 0.0
        
        # POD (Probability of Detection, 命中率)
        if (tp + fn) > 0:
            metrics['POD'] = tp / (tp + fn)
        else:
            metrics['POD'] = 0.0
        
        # FAR (False Alarm Ratio, 虚警率)
        if (tp + fp) > 0:
            metrics['FAR'] = fp / (tp + fp)
        else:
            metrics['FAR'] = 0.0
        
        # Accuracy
        if (tp + tn + fp + fn) > 0:
            metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        else:
            metrics['Accuracy'] = 0.0
        
        # F1 Score
        if (metrics['POD'] + (1 - metrics['FAR'])) > 0:
            metrics['F1'] = 2 * (metrics['POD'] * (1 - metrics['FAR'])) / (metrics['POD'] + (1 - metrics['FAR']))
        else:
            metrics['F1'] = 0.0
        
        return metrics
    
    def calculate_image_metrics(self, y_true, y_pred):
        """计算图像质量指标：MSE, MAE, SSIM"""
        metrics = {}
        
        # MSE (Mean Squared Error)
        metrics['MSE'] = np.mean((y_true - y_pred) ** 2)
        
        # MAE (Mean Absolute Error)
        metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
        
        # RMSE (Root Mean Squared Error)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        
        # PSNR (Peak Signal-to-Noise Ratio)
        if metrics['MSE'] > 0:
            max_val = 1.0
            metrics['PSNR'] = 20 * np.log10(max_val / np.sqrt(metrics['MSE']))
        else:
            metrics['PSNR'] = float('inf')
        
        # SSIM (Structural Similarity Index)
        metrics['SSIM'] = self._calculate_ssim(y_true, y_pred)
        
        # Correlation Coefficient
        corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        metrics['Correlation'] = corr if not np.isnan(corr) else 0.0
        
        return metrics
    
    def _calculate_ssim(self, img1, img2, window_size=11):
        """计算SSIM"""
        from scipy.ndimage import uniform_filter
        
        # 常数
        K1 = 0.01
        K2 = 0.03
        L = 1.0  # 动态范围
        
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        
        # 均值
        mu1 = uniform_filter(img1, window_size)
        mu2 = uniform_filter(img2, window_size)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = uniform_filter(img1 * img1, window_size) - mu1_sq
        sigma2_sq = uniform_filter(img2 * img2, window_size) - mu2_sq
        sigma12 = uniform_filter(img1 * img2, window_size) - mu1_mu2
        
        # SSIM公式
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    def calculate_all_metrics(self, y_true, y_pred):
        """计算所有指标"""
        all_metrics = {}
        
        # 图像质量指标
        img_metrics = self.calculate_image_metrics(y_true, y_pred)
        all_metrics.update(img_metrics)
        
        # 分类指标（按阈值）
        classification_metrics = {}
        for threshold in self.thresholds:
            cls_metrics = self.calculate_classification_metrics(y_true, y_pred, threshold)
            classification_metrics[f'{threshold}dBZ'] = cls_metrics
        
        all_metrics['classification'] = classification_metrics
        
        return all_metrics

# ================= 4. 评估器类 =================
class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.metrics_calculator = ComprehensiveMetrics(thresholds=config['dbz_thresholds'])
    
    def evaluate_model(self, model, data_loader, model_name):
        """评估单个模型"""
        print(f"\n评估模型: {model_name}")
        
        model.eval()
        
        # 存储指标
        all_image_metrics = []
        all_classification_metrics = {f'{t}dBZ': [] for t in self.config['dbz_thresholds']}
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc=f"处理 {model_name}")):
                if sample_count >= self.config['max_samples']:
                    break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 预测
                outputs = model(inputs)
                
                # 转换为numpy
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                batch_size = outputs_np.shape[0]
                
                # 处理每个样本
                for i in range(batch_size):
                    if sample_count >= self.config['max_samples']:
                        break
                    
                    # 取预测的第一帧和真实的第一帧
                    pred_frame = outputs_np[i, 0, 0]  # [batch, frame, channel, height, width]
                    true_frame = targets_np[i, 0, 0]
                    
                    # 计算所有指标
                    metrics = self.metrics_calculator.calculate_all_metrics(true_frame, pred_frame)
                    
                    # 收集图像质量指标
                    img_metrics = {k: v for k, v in metrics.items() if k != 'classification'}
                    all_image_metrics.append(img_metrics)
                    
                    # 收集分类指标
                    for threshold_key, cls_metrics in metrics['classification'].items():
                        all_classification_metrics[threshold_key].append(cls_metrics)
                    
                    sample_count += 1
                
                # 清理内存
                del inputs, targets, outputs, outputs_np, targets_np
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # 计算平均指标
        avg_metrics = self._average_metrics(all_image_metrics, all_classification_metrics)
        
        print(f"  完成评估 {sample_count} 个样本")
        print(f"  MSE: {avg_metrics.get('MSE', 0):.6f}")
        print(f"  MAE: {avg_metrics.get('MAE', 0):.6f}")
        print(f"  SSIM: {avg_metrics.get('SSIM', 0):.4f}")
        
        # 显示主要分类指标（以25dBZ为例）
        if 'classification' in avg_metrics and '25dBZ' in avg_metrics['classification']:
            cls_25 = avg_metrics['classification']['25dBZ']
            print(f"  CSI@25dBZ: {cls_25.get('CSI', 0):.4f}")
            print(f"  POD@25dBZ: {cls_25.get('POD', 0):.4f}")
            print(f"  FAR@25dBZ: {cls_25.get('FAR', 0):.4f}")
        
        return avg_metrics, sample_count
    
    def _average_metrics(self, image_metrics_list, classification_metrics_dict):
        """计算平均指标"""
        avg_metrics = {}
        
        # 平均图像质量指标
        if image_metrics_list:
            for key in image_metrics_list[0].keys():
                values = [m[key] for m in image_metrics_list]
                avg_metrics[key] = np.mean(values)
        
        # 平均分类指标
        avg_classification = {}
        for threshold_key, metrics_list in classification_metrics_dict.items():
            if metrics_list:
                avg_cls = {}
                for metric_name in metrics_list[0].keys():
                    values = [m[metric_name] for m in metrics_list]
                    avg_cls[metric_name] = np.mean(values)
                avg_classification[threshold_key] = avg_cls
        
        avg_metrics['classification'] = avg_classification
        
        return avg_metrics

# ================= 5. 可视化函数 =================
def plot_results(results_dict, output_dir):
    """绘制结果图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    models = list(results_dict.keys())
    
    # 1. 主要指标对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('MSE', '均方误差 (MSE)'),
        ('MAE', '平均绝对误差 (MAE)'), 
        ('SSIM', '结构相似性指数 (SSIM)'),
        ('RMSE', '均方根误差 (RMSE)'),
        ('PSNR', '峰值信噪比 (PSNR)'),
        ('Correlation', '相关系数')
    ]
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        values = []
        for model in models:
            if metric_key in results_dict[model]:
                values.append(results_dict[model][metric_key])
            else:
                values.append(0)
        
        bars = ax.bar(models, values, alpha=0.7, color=['#1f77b4', '#ff7f0e'][:len(models)])
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom')
        
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('模型性能对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()
    
    print(f"✓ 指标对比图保存至: {os.path.join(output_dir, 'metrics_comparison.png')}")
    
    # 2. CSI阈值曲线
    if len(models) > 0 and 'classification' in results_dict[models[0]]:
        thresholds = EVAL_CONFIG['dbz_thresholds']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # CSI曲线
        ax = axes[0, 0]
        for model_name in models:
            if 'classification' in results_dict[model_name]:
                csi_values = []
                for t in thresholds:
                    key = f'{t}dBZ'
                    if key in results_dict[model_name]['classification']:
                        csi_values.append(results_dict[model_name]['classification'][key]['CSI'])
                    else:
                        csi_values.append(0)
                ax.plot(thresholds, csi_values, marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('阈值 (dBZ)')
        ax.set_ylabel('CSI')
        ax.set_title('CSI vs 阈值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # POD-FAR曲线
        ax = axes[0, 1]
        for model_name in models:
            if 'classification' in results_dict[model_name]:
                pod_values = []
                far_values = []
                for t in thresholds:
                    key = f'{t}dBZ'
                    if key in results_dict[model_name]['classification']:
                        pod_values.append(results_dict[model_name]['classification'][key]['POD'])
                        far_values.append(results_dict[model_name]['classification'][key]['FAR'])
                if pod_values and far_values:
                    ax.plot(far_values, pod_values, marker='s', label=model_name, linewidth=2)
        
        ax.set_xlabel('虚警率 (FAR)')
        ax.set_ylabel('命中率 (POD)')
        ax.set_title('POD-FAR曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # F1曲线
        ax = axes[1, 0]
        for model_name in models:
            if 'classification' in results_dict[model_name]:
                f1_values = []
                for t in thresholds:
                    key = f'{t}dBZ'
                    if key in results_dict[model_name]['classification']:
                        f1_values.append(results_dict[model_name]['classification'][key]['F1'])
                    else:
                        f1_values.append(0)
                ax.plot(thresholds, f1_values, marker='^', label=model_name, linewidth=2)
        
        ax.set_xlabel('阈值 (dBZ)')
        ax.set_ylabel('F1分数')
        ax.set_title('F1分数 vs 阈值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax = axes[1, 1]
        for model_name in models:
            if 'classification' in results_dict[model_name]:
                acc_values = []
                for t in thresholds:
                    key = f'{t}dBZ'
                    if key in results_dict[model_name]['classification']:
                        acc_values.append(results_dict[model_name]['classification'][key]['Accuracy'])
                    else:
                        acc_values.append(0)
                ax.plot(thresholds, acc_values, marker='d', label=model_name, linewidth=2)
        
        ax.set_xlabel('阈值 (dBZ)')
        ax.set_ylabel('准确率')
        ax.set_title('准确率 vs 阈值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('分类指标阈值曲线', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_curves.png'), dpi=300)
        plt.close()
        
        print(f"✓ 阈值曲线图保存至: {os.path.join(output_dir, 'threshold_curves.png')}")

# ================= 6. 保存结果函数 =================
def save_results(results_dict, output_dir, sample_count):
    """保存评估结果"""
    # 保存JSON
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    
    # 转换numpy类型
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    results_converted = convert_types(results_dict)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)
    
    # 保存文本报告
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("雷达回波预测模型综合评估报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评估样本数: {sample_count}\n")
        f.write(f"Batch大小: {EVAL_CONFIG['batch_size']}\n")
        f.write(f"设备: {EVAL_CONFIG['device']}\n\n")
        
        f.write("模型性能汇总:\n")
        f.write("-" * 70 + "\n\n")
        
        for model_name, metrics in results_dict.items():
            f.write(f"{model_name}:\n")
            
            f.write("\n  图像质量指标:\n")
            for key in ['MSE', 'MAE', 'RMSE', 'SSIM', 'PSNR', 'Correlation']:
                if key in metrics:
                    f.write(f"    {key}: {metrics[key]:.6f}\n")
            
            f.write("\n  分类指标 (以25dBZ为例):\n")
            if 'classification' in metrics and '25dBZ' in metrics['classification']:
                cls_25 = metrics['classification']['25dBZ']
                for key in ['CSI', 'POD', 'FAR', 'Accuracy', 'F1']:
                    if key in cls_25:
                        f.write(f"    {key}: {cls_25[key]:.4f}\n")
            
            f.write("\n")
    
    print(f"✓ JSON结果保存至: {json_path}")
    print(f"✓ 文本报告保存至: {report_path}")

# ================= 7. 主评估函数 =================
def main():
    """主评估函数"""
    
    # 1. 加载数据集
    print("\n" + "=" * 70)
    print("1. 加载数据集")
    print("=" * 70)
    
    try:
        dataset = RadarLazyDataset(EVAL_CONFIG['images_dir'], EVAL_CONFIG['list_file'])
        print(f"✓ 数据集加载成功")
        print(f"  数据集大小: {len(dataset)} 个序列")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        return
    
    # 创建数据加载器
    from torch.utils.data import DataLoader, Subset
    
    max_samples = min(EVAL_CONFIG['max_samples'], len(dataset))
    indices = list(range(max_samples))
    subset = Subset(dataset, indices)
    
    data_loader = DataLoader(
        subset,
        batch_size=EVAL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✓ 数据加载器创建成功")
    print(f"  评估样本数: {max_samples}")
    print(f"  Batch大小: {EVAL_CONFIG['batch_size']}")
    
    # 2. 初始化评估器
    evaluator = ModelEvaluator(EVAL_CONFIG)
    
    # 3. 加载并评估模型
    print("\n" + "=" * 70)
    print("2. 加载并评估模型")
    print("=" * 70)
    
    results = {}
    
    for model_name, model_path in EVAL_CONFIG['model_paths'].items():
        if not os.path.exists(model_path):
            print(f"✗ {model_name}: 模型文件不存在 - {model_path}")
            continue
        
        try:
            # 加载模型
            if model_name == 'LSTM-CNN':
                model = LSTMCNN(
                    input_channels=1,
                    encoder_dims=[8, 16, 32],
                    convlstm_dims=[32],
                    decoder_dims=[32, 16, 8],
                    kernel_size=(3, 3)
                )
            elif model_name == 'ViT':
                model = SimpleViT(
                    img_size=EVAL_CONFIG['img_size'][0],
                    patch_size=16,
                    in_channels=1,
                    embed_dim=256,
                    num_heads=4,
                    depth=3,
                    out_channels=1,
                    future_frames=EVAL_CONFIG['target_len'],
                    dropout=0.1
                )
            else:
                continue
            
            # 加载权重
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.to(evaluator.device)
            model.eval()
            
            print(f"✓ {model_name}: 加载成功")
            
            # 评估模型
            metrics, sample_count = evaluator.evaluate_model(model, data_loader, model_name)
            results[model_name] = metrics
            
        except Exception as e:
            print(f"✗ {model_name}: 加载或评估失败 - {e}")
    
    if not results:
        print("✗ 没有模型成功加载和评估")
        return
    
    # 4. 可视化结果
    print("\n" + "=" * 70)
    print("3. 生成可视化结果")
    print("=" * 70)
    
    plot_results(results, EVAL_CONFIG['output_dir'])
    
    # 5. 保存结果
    print("\n" + "=" * 70)
    print("4. 保存评估结果")
    print("=" * 70)
    
    save_results(results, EVAL_CONFIG['output_dir'], max_samples)
    
    # 6. 显示总结
    print("\n" + "=" * 70)
    print("评估总结")
    print("=" * 70)
    
    print(f"\n评估完成!")
    print(f"评估了 {len(results)} 个模型")
    print(f"每个模型评估了 {max_samples} 个样本")
    
    if len(results) > 1:
        # 找出最佳模型
        best_mse = min(results.items(), key=lambda x: x[1].get('MSE', float('inf')))
        best_ssim = max(results.items(), key=lambda x: x[1].get('SSIM', 0))
        
        print(f"\n最佳MSE模型: {best_mse[0]} (MSE={best_mse[1].get('MSE', 0):.6f})")
        print(f"最佳SSIM模型: {best_ssim[0]} (SSIM={best_ssim[1].get('SSIM', 0):.4f})")
        
        # CSI对比
        csi_scores = []
        for model_name, metrics in results.items():
            if 'classification' in metrics and '25dBZ' in metrics['classification']:
                csi = metrics['classification']['25dBZ'].get('CSI', 0)
                csi_scores.append((model_name, csi))
        
        if csi_scores:
            best_csi = max(csi_scores, key=lambda x: x[1])
            worst_csi = min(csi_scores, key=lambda x: x[1])
            print(f"最佳CSI模型: {best_csi[0]} (CSI={best_csi[1]:.4f})")
            print(f"最差CSI模型: {worst_csi[0]} (CSI={worst_csi[1]:.4f})")
    
    print(f"\n所有结果保存在: {EVAL_CONFIG['output_dir']}")

# ================= 8. 执行主函数 =================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n评估被用户中断")
    except Exception as e:
        print(f"\n评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()