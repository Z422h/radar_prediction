"""
修复版结果可视化与案例分析系统
处理图像尺寸不一致的问题
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# ================= 设置路径 =================
PROJECT_ROOT = r"D:\radar_prediction\codes"
sys.path.insert(0, PROJECT_ROOT)

print("=" * 70)
print("Results Visualization System (Fixed Version)")
print("=" * 70)

# ================= 导入模块 =================
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'model_design'))
    import lstm_cnn_model
    
    # 尝试导入train模块，如果失败则使用简化版本
    try:
        import train_lstm_cnn_model
        RadarLazyDataset = train_lstm_cnn_model.RadarLazyDataset
        print("✓ Successfully imported RadarLazyDataset")
    except:
        # 创建简化版本
        class RadarLazyDataset:
            def __init__(self, root_dir, list_file):
                self.root_dir = root_dir
                self.sequences = []
                if os.path.exists(list_file):
                    with open(list_file, 'r') as f:
                        for line in f:
                            self.sequences.append(line.strip().split(','))
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                import cv2
                file_names = self.sequences[idx]
                frames = []
                
                for fname in file_names:
                    path = os.path.join(self.root_dir, fname)
                    if os.path.exists(path):
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # 修复：统一调整到256x256
                            img = cv2.resize(img, (256, 256))
                            frames.append(img / 255.0)
                        else:
                            frames.append(np.zeros((256, 256), dtype=np.float32))
                    else:
                        frames.append(np.zeros((256, 256), dtype=np.float32))
                
                frames = np.array(frames, dtype=np.float32)
                frames = np.expand_dims(frames, axis=1)
                
                input_seq = torch.from_numpy(frames[:10])
                target_seq = torch.from_numpy(frames[10:])
                
                return input_seq, target_seq
        print("✓ Using simplified dataset class")
    
    LSTMCNN = lstm_cnn_model.LSTMCNN
    SimpleViT = lstm_cnn_model.SimpleViT
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# ================= 修复版配置 =================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'images_dir': os.path.join(PROJECT_ROOT, 'processed_data', 'images'),
    'list_file': os.path.join(PROJECT_ROOT, 'processed_data', 'train_list.txt'),
    'output_dir': os.path.join(PROJECT_ROOT, 'visualization_results_fixed'),
    'model_paths': {
        'LSTM-CNN': os.path.join(PROJECT_ROOT, 'checkpoints', 'best_lstm_cnn_model.pth'),
        'ViT': os.path.join(PROJECT_ROOT, 'checkpoints', 'best_vit_model.pth'),
    },
    'img_size': (256, 256),  # 固定图像尺寸
    'seq_len': 20,
    'input_len': 10,
    'target_len': 10,
}

# 创建输出目录
os.makedirs(CONFIG['output_dir'], exist_ok=True)
for subdir in ['gifs', 'comparisons']:
    os.makedirs(os.path.join(CONFIG['output_dir'], subdir), exist_ok=True)

print(f"\nConfiguration:")
print(f"  Device: {CONFIG['device']}")
print(f"  Output directory: {CONFIG['output_dir']}")
print(f"  Image size: {CONFIG['img_size']}")

# ================= 修复版数据集加载器 =================
class FixedRadarDataset:
    """Fixed dataset ensuring consistent image size"""
    
    def __init__(self, images_dir, list_file, img_size=(256, 256)):
        self.images_dir = images_dir
        self.img_size = img_size
        self.sequences = []
        
        if os.path.exists(list_file):
            with open(list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.sequences.append(line.strip().split(','))
            print(f"✓ Loaded {len(self.sequences)} sequences")
        else:
            print(f"✗ List file does not exist: {list_file}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        file_names = self.sequences[idx]
        frames = []
        
        for fname in file_names:
            path = os.path.join(self.images_dir, fname)
            if os.path.exists(path):
                try:
                    # 读取并确保尺寸正确
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # 修复：强制调整到指定尺寸
                        if img.shape != self.img_size:
                            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
                        frames.append(img / 255.0)
                    else:
                        frames.append(np.zeros(self.img_size, dtype=np.float32))
                except Exception as e:
                    print(f"  Warning: Error processing image {path}: {e}")
                    frames.append(np.zeros(self.img_size, dtype=np.float32))
            else:
                frames.append(np.zeros(self.img_size, dtype=np.float32))
        
        # 转换为numpy数组
        frames = np.array(frames, dtype=np.float32)
        
        # 添加通道维度
        frames = np.expand_dims(frames, axis=1)
        
        # 分割输入和目标
        input_seq = torch.from_numpy(frames[:CONFIG['input_len']])
        target_seq = torch.from_numpy(frames[CONFIG['input_len']:])
        
        return input_seq, target_seq

# ================= 修复版动态可视化生成器 =================
class FixedDynamicVisualizer:
    """Fixed dynamic visualization generator"""
    
    def __init__(self, config):
        self.config = config
        
    def create_comparison_gif(self, inputs, predictions, targets, sample_idx, model_name, save_path):
        """Create comparison GIF"""
        
        try:
            # 确保所有数组形状一致
            assert inputs.shape[1:] == predictions.shape[1:] == targets.shape[1:], \
                f"Shape mismatch: inputs {inputs.shape}, pred {predictions.shape}, target {targets.shape}"
            
            # 转换为0-255范围
            inputs_uint8 = (inputs * 255).astype(np.uint8)
            predictions_uint8 = (predictions * 255).astype(np.uint8)
            targets_uint8 = (targets * 255).astype(np.uint8)
            
            # 准备帧列表
            frames = []
            
            # 生成对比帧
            for t in range(min(len(predictions), len(targets))):
                # 获取输入帧（如果是前几帧）
                if t < len(inputs):
                    input_frame = self._apply_colormap(inputs_uint8[t])
                else:
                    input_frame = np.zeros((self.config['img_size'][0], self.config['img_size'][1], 3), dtype=np.uint8)
                
                # 预测帧和真实帧
                pred_frame = self._apply_colormap(predictions_uint8[t])
                target_frame = self._apply_colormap(targets_uint8[t])
                
                # 添加标签
                cv2.putText(input_frame, f"Input", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(pred_frame, f"Pred t={t+1}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(target_frame, f"True t={t+1}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 水平拼接
                combined = np.hstack([input_frame, pred_frame, target_frame])
                frames.append(combined)
            
            # 保存为GIF
            if frames:
                imageio.mimsave(save_path, frames, fps=2)
                print(f"  ✓ GIF saved to: {save_path}")
                return True
            else:
                print(f"  ✗ Cannot generate GIF: No frames")
                return False
                
        except Exception as e:
            print(f"  ✗ Failed to create GIF: {e}")
            return False
    
    def _apply_colormap(self, img):
        """Apply color map"""
        if len(img.shape) == 2:
            return cv2.applyColorMap(img, cv2.COLORMAP_JET)
        else:
            # 如果是单通道但维度不对
            if img.shape[0] == 1:
                img = img[0]
            return cv2.applyColorMap(img, cv2.COLORMAP_JET)

# ================= 修复版模型加载器 =================
class FixedModelLoader:
    """Fixed model loader"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
    def load_model(self, model_name, model_path):
        """Load single model"""
        if not os.path.exists(model_path):
            print(f"  ✗ {model_name}: File does not exist")
            return None
        
        try:
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
                    img_size=self.config['img_size'][0],
                    patch_size=16,
                    in_channels=1,
                    embed_dim=256,
                    num_heads=4,
                    depth=3,
                    out_channels=1,
                    future_frames=self.config['target_len'],
                    dropout=0.1
                )
            else:
                return None
            
            # 加载权重
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.to(self.device)
            model.eval()
            
            print(f"  ✓ {model_name}: Loaded successfully")
            return model
            
        except Exception as e:
            print(f"  ✗ {model_name}: Loading failed - {e}")
            return None

# ================= 简化版案例分析 =================
def analyze_sample_predictions(model, dataset, sample_indices, model_name, output_dir):
    """Analyze sample predictions"""
    
    print(f"\nAnalyzing model: {model_name}")
    
    visualizer = FixedDynamicVisualizer(CONFIG)
    results = {
        'gifs_created': 0,
        'samples_analyzed': 0
    }
    
    for i, sample_idx in enumerate(sample_indices):
        print(f"  Sample {i+1}/{len(sample_indices)} (index: {sample_idx})")
        
        try:
            # 获取数据
            inputs, targets = dataset[sample_idx]
            
            # 添加batch维度
            inputs = inputs.unsqueeze(0).to(CONFIG['device'])
            
            # 预测
            with torch.no_grad():
                predictions = model(inputs)
            
            # 转换为numpy并确保形状正确
            inputs_np = inputs.cpu().numpy()[0]  # [frames, 1, H, W]
            predictions_np = predictions.cpu().numpy()[0]
            targets_np = targets.numpy()
            
            # 移除通道维度
            if inputs_np.ndim == 4 and inputs_np.shape[1] == 1:
                inputs_np = inputs_np[:, 0, :, :]  # [frames, H, W]
                predictions_np = predictions_np[:, 0, :, :]
                targets_np = targets_np[:, 0, :, :]
            
            # 检查形状
            assert inputs_np.shape[1:] == (256, 256), f"Input shape error: {inputs_np.shape}"
            assert predictions_np.shape[1:] == (256, 256), f"Prediction shape error: {predictions_np.shape}"
            assert targets_np.shape[1:] == (256, 256), f"Target shape error: {targets_np.shape}"
            
            # 生成GIF
            gif_dir = os.path.join(output_dir, 'gifs', model_name)
            os.makedirs(gif_dir, exist_ok=True)
            
            gif_path = os.path.join(gif_dir, f'sample_{sample_idx}.gif')
            
            # 使用最后5帧输入和前5帧预测/真实
            success = visualizer.create_comparison_gif(
                inputs_np[-5:],  # Last 5 input frames
                predictions_np[:5],  # First 5 prediction frames
                targets_np[:5],  # First 5 target frames
                sample_idx,
                model_name,
                gif_path
            )
            
            if success:
                results['gifs_created'] += 1
            
            # 创建静态对比图
            create_static_comparison(
                inputs_np, predictions_np, targets_np,
                sample_idx, model_name, output_dir
            )
            
            results['samples_analyzed'] += 1
            
        except Exception as e:
            print(f"    ✗ Processing failed: {e}")
            continue
    
    return results

def create_static_comparison(inputs, predictions, targets, sample_idx, model_name, output_dir):
    """Create static comparison chart (English version)"""
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Last input frame
        ax = axes[0, 0]
        im = ax.imshow(inputs[-1], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Input (Last Frame)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        # First prediction frame
        ax = axes[0, 1]
        im = ax.imshow(predictions[0], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Prediction (First Frame)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        # First target frame
        ax = axes[0, 2]
        im = ax.imshow(targets[0], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Ground Truth (First Frame)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        # Last prediction frame
        ax = axes[1, 0]
        im = ax.imshow(predictions[-1], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Prediction (Last Frame)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        # Last target frame
        ax = axes[1, 1]
        im = ax.imshow(targets[-1], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Ground Truth (Last Frame)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        # Error map (first frame)
        ax = axes[1, 2]
        error = np.abs(predictions[0] - targets[0])
        im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.5)
        ax.set_title('Absolute Error (First Frame)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Model Comparison: {model_name} - Sample {sample_idx}', fontsize=16)
        plt.tight_layout()
        
        # 保存图像
        img_dir = os.path.join(output_dir, 'comparisons', model_name)
        os.makedirs(img_dir, exist_ok=True)
        
        img_path = os.path.join(img_dir, f'sample_{sample_idx}_comparison.png')
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Comparison chart saved to: {img_path}")
        
    except Exception as e:
        print(f"    ✗ Failed to create comparison chart: {e}")

# ================= 主分析函数 =================
def perform_fixed_analysis():
    """Execute fixed version analysis"""
    
    print("\n" + "=" * 70)
    print("Starting Fixed Version Analysis")
    print("=" * 70)
    
    # 1. 加载修复版数据集
    print("\n1. Loading dataset...")
    try:
        dataset = FixedRadarDataset(
            CONFIG['images_dir'],
            CONFIG['list_file'],
            CONFIG['img_size']
        )
        print(f"  ✓ Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        return
    
    # 2. 加载模型
    print("\n2. Loading models...")
    model_loader = FixedModelLoader(CONFIG)
    
    models = {}
    for model_name, model_path in CONFIG['model_paths'].items():
        model = model_loader.load_model(model_name, model_path)
        if model is not None:
            models[model_name] = model
    
    if not models:
        print("  ✗ No models successfully loaded")
        return
    
    # 3. 选择样本进行分析
    print("\n3. Selecting analysis samples...")
    
    import random
    # 选择一些样本（确保在数据集范围内）
    max_samples = min(20, len(dataset))
    sample_indices = random.sample(range(max_samples), min(5, max_samples))
    
    print(f"  Analysis sample indices: {sample_indices}")
    
    # 4. 分析每个模型
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name}...")
        
        results = analyze_sample_predictions(
            model, dataset, sample_indices, model_name, CONFIG['output_dir']
        )
        
        all_results[model_name] = results
    
    # 5. 生成报告
    print("\n4. Generating analysis report...")
    generate_fixed_report(all_results, CONFIG['output_dir'])
    
    print("\n" + "=" * 70)
    print("Fixed Version Analysis Completed!")
    print("=" * 70)

def generate_fixed_report(all_results, output_dir):
    """Generate fixed version report"""
    
    report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_size': CONFIG['img_size'],
        'models_analyzed': list(all_results.keys()),
        'results': all_results
    }
    
    # 保存JSON报告
    report_path = os.path.join(output_dir, 'analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成文本摘要
    txt_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Radar Echo Prediction Model Visualization Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Analysis Time: {report['analysis_date']}\n")
        f.write(f"Image Size: {report['image_size']}\n")
        f.write(f"Models Analyzed: {', '.join(report['models_analyzed'])}\n\n")
        
        f.write("Analysis Results:\n")
        f.write("-" * 70 + "\n")
        
        for model_name, results in report['results'].items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Samples Analyzed: {results['samples_analyzed']}\n")
            f.write(f"  GIFs Created: {results['gifs_created']}\n")
        
        f.write("\nOutput Files:\n")
        f.write("-" * 70 + "\n")
        
        # 列出输出文件
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # 只列出前10个文件
                f.write(f"{subindent}{file}\n")
            if len(files) > 10:
                f.write(f"{subindent}... and {len(files) - 10} more files\n")
        
        f.write("\nUsage Instructions:\n")
        f.write("-" * 70 + "\n")
        f.write("1. GIF files are in the gifs/ directory\n")
        f.write("2. Static comparison charts are in the comparisons/ directory\n")
        f.write("3. Detailed report is in analysis_report.json\n")
    
    print(f"  ✓ Report saved to: {report_path}")
    print(f"  ✓ Summary saved to: {txt_path}")

# ================= 执行主函数 =================
if __name__ == "__main__":
    try:
        perform_fixed_analysis()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()