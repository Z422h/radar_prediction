import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import imageio
import glob
import sys

# 导入模型
from lstm_cnn_model import LSTMCNN, SimpleViT

# ================= 配置区域 =================
STORM_CONFIG = {
    'storm_start': '20250625_180000',             # 强对流开始时间
    'storm_end': '20250625_230000',               # 强对流结束时间
    'output_dir': 'storm_analysis_20250625',    # 分析结果保存目录
    'img_size': (256, 256),                       # 图像尺寸
    'seq_len': 20,                                # 序列长度
    'input_len': 10,                              # 输入长度
    'target_len': 10,                             # 目标长度
    'batch_size': 1,                              # 批大小
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'storm_name': 'Strong_Convection_20250625'
}

# ================= 首先查找文件 =================
def find_project_files():
    """
    查找项目中的所有必要文件
    """
    print("\nSearching for project files...")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current script directory: {current_dir}")
    
    # 查找可能的根目录
    possible_roots = [
        current_dir,  # model_design目录
        os.path.dirname(current_dir),  # codes目录
        os.path.dirname(os.path.dirname(current_dir)),  # radar_prediction目录
    ]
    
    # 搜索processed_data目录
    processed_data_dirs = []
    images_dirs = []
    
    for root in possible_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if 'processed_data' in dirpath:
                processed_data_dirs.append(dirpath)
                # 检查是否有images子目录
                images_path = os.path.join(dirpath, 'images')
                if os.path.exists(images_path):
                    images_dirs.append(images_path)
            
            # 直接搜索images目录
            if 'images' in dirnames:
                images_path = os.path.join(dirpath, 'images')
                images_dirs.append(images_path)
    
    # 去重
    processed_data_dirs = list(set(processed_data_dirs))
    images_dirs = list(set(images_dirs))
    
    print(f"\nFound {len(processed_data_dirs)} processed_data directories:")
    for dir in processed_data_dirs[:5]:  # 只显示前5个
        print(f"  - {dir}")
    
    print(f"\nFound {len(images_dirs)} images directories:")
    for dir in images_dirs[:5]:  # 只显示前5个
        print(f"  - {dir}")
    
    # 查找train_list.txt文件
    train_list_files = []
    for dir in processed_data_dirs:
        possible_files = [
            os.path.join(dir, 'train_list.txt'),
            os.path.join(dir, 'list.txt'),
            os.path.join(dir, 'sequence_list.txt'),
        ]
        
        for file in possible_files:
            if os.path.exists(file):
                train_list_files.append(file)
    
    # 在整个项目中搜索train_list.txt
    if not train_list_files:
        for root in possible_roots:
            for dirpath, dirnames, filenames in os.walk(root):
                for filename in filenames:
                    if 'train_list' in filename or 'list.txt' in filename:
                        file_path = os.path.join(dirpath, filename)
                        train_list_files.append(file_path)
    
    print(f"\nFound {len(train_list_files)} train list files:")
    for file in train_list_files[:5]:  # 只显示前5个
        print(f"  - {file}")
    
    # 查找模型文件
    checkpoint_dirs = []
    model_files = []
    
    for root in possible_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if 'checkpoints' in dirpath:
                checkpoint_dirs.append(dirpath)
                # 查找模型文件
                for filename in filenames:
                    if filename.endswith('.pth'):
                        model_files.append(os.path.join(dirpath, filename))
    
    # 去重
    checkpoint_dirs = list(set(checkpoint_dirs))
    model_files = list(set(model_files))
    
    print(f"\nFound {len(checkpoint_dirs)} checkpoint directories:")
    for dir in checkpoint_dirs[:3]:
        print(f"  - {dir}")
    
    print(f"\nFound {len(model_files)} model files:")
    for file in model_files[:10]:  # 只显示前10个
        print(f"  - {file}")
    
    # 选择文件
    selected_files = {}
    
    # 1. 选择train_list.txt
    if train_list_files:
        # 优先选择在processed_data目录中的文件
        for file in train_list_files:
            if 'processed_data' in file:
                selected_files['list_file'] = file
                break
        if 'list_file' not in selected_files:
            selected_files['list_file'] = train_list_files[0]
    else:
        print("\nWARNING: No train_list.txt found!")
    
    # 2. 选择images目录
    if images_dirs:
        # 优先选择在processed_data目录中的images
        for dir in images_dirs:
            if 'processed_data' in dir:
                selected_files['images_dir'] = dir
                break
        if 'images_dir' not in selected_files:
            selected_files['images_dir'] = images_dirs[0]
    else:
        print("\nWARNING: No images directory found!")
    
    # 3. 选择模型文件
    model_paths = {}
    
    # 查找LSTM-CNN模型
    lstm_files = [f for f in model_files if 'lstm' in f.lower()]
    if lstm_files:
        # 优先选择best_lstm_cnn_model.pth
        for file in lstm_files:
            if 'best' in file.lower():
                model_paths['lstm_cnn'] = file
                break
        if 'lstm_cnn' not in model_paths:
            model_paths['lstm_cnn'] = lstm_files[0]
    
    # 查找ViT模型
    vit_files = [f for f in model_files if 'vit' in f.lower()]
    if vit_files:
        # 优先选择best_vit_model.pth
        for file in vit_files:
            if 'best' in file.lower():
                model_paths['vit'] = file
                break
        if 'vit' not in model_paths:
            model_paths['vit'] = vit_files[0]
    
    selected_files['model_paths'] = model_paths
    
    return selected_files

def check_files_exist(selected_files):
    """
    检查选中的文件是否存在
    """
    print("\n" + "="*60)
    print("FILE CHECK RESULTS")
    print("="*60)
    
    all_exist = True
    
    # 检查train_list.txt
    if 'list_file' in selected_files:
        list_file = selected_files['list_file']
        if os.path.exists(list_file):
            # 统计行数
            try:
                with open(list_file, 'r') as f:
                    lines = f.readlines()
                print(f"✓ train_list.txt: {list_file} ({len(lines)} sequences)")
            except:
                print(f"✓ train_list.txt: {list_file}")
        else:
            print(f"✗ train_list.txt: {list_file} (NOT FOUND)")
            all_exist = False
    else:
        print("✗ train_list.txt: NOT SELECTED")
        all_exist = False
    
    # 检查images目录
    if 'images_dir' in selected_files:
        images_dir = selected_files['images_dir']
        if os.path.exists(images_dir):
            # 统计图片数量
            png_files = glob.glob(os.path.join(images_dir, "*.png"))
            print(f"✓ images directory: {images_dir} ({len(png_files)} PNG files)")
        else:
            print(f"✗ images directory: {images_dir} (NOT FOUND)")
            all_exist = False
    else:
        print("✗ images directory: NOT SELECTED")
        all_exist = False
    
    # 检查模型文件
    if 'model_paths' in selected_files:
        model_paths = selected_files['model_paths']
        
        for model_type, model_path in model_paths.items():
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024*1024)  # MB
                print(f"✓ {model_type} model: {model_path} ({file_size:.1f} MB)")
            else:
                print(f"✗ {model_type} model: {model_path} (NOT FOUND)")
                all_exist = False
    
    print("="*60)
    
    return all_exist

# ================= 工具函数 =================
def load_storm_sequences(images_dir, list_file):
    """
    加载包含强对流时段的序列
    """
    # 解析时间
    start_dt = datetime.strptime(STORM_CONFIG['storm_start'], "%Y%m%d_%H%M%S")
    end_dt = datetime.strptime(STORM_CONFIG['storm_end'], "%Y%m%d_%H%M%S")
    
    storm_sequences = []
    storm_indices = []
    
    try:
        # 加载序列文件
        with open(list_file, 'r') as f:
            sequences = [line.strip().split(',') for line in f]
        
        print(f"\nTotal sequences found in list: {len(sequences)}")
        print(f"Looking for storm sequences between {STORM_CONFIG['storm_start']} and {STORM_CONFIG['storm_end']}")
        
        # 查找包含风暴时段的序列
        for idx, seq_filenames in enumerate(sequences):
            # 获取序列的时间范围
            try:
                seq_start_str = seq_filenames[0].split('.')[0]
                seq_end_str = seq_filenames[-1].split('.')[0]
                
                # 处理插值帧的后缀
                if '_interp' in seq_start_str:
                    seq_start_str = seq_start_str.replace('_interp', '')
                if '_interp' in seq_end_str:
                    seq_end_str = seq_end_str.replace('_interp', '')
                
                seq_start = datetime.strptime(seq_start_str, "%Y%m%d_%H%M%S")
                seq_end = datetime.strptime(seq_end_str, "%Y%m%d_%H%M%S")
                
                # 检查是否与风暴时段有重叠
                if not (seq_end < start_dt or seq_start > end_dt):
                    storm_indices.append(idx)
                    storm_sequences.append(seq_filenames)
                    print(f"  Found storm sequence {idx}: {seq_start} to {seq_end}")
            except ValueError as e:
                # 跳过格式错误的文件名
                continue
                
    except Exception as e:
        print(f"Error loading sequences: {e}")
        return [], []
    
    print(f"\nFound {len(storm_indices)} storm sequences")
    
    # 如果没有找到序列，尝试直接搜索文件
    if len(storm_indices) == 0:
        print("\nTrying to find storm images directly...")
        storm_sequences = find_storm_images_directly(images_dir, start_dt, end_dt)
    
    return storm_indices, storm_sequences

def find_storm_images_directly(images_dir, start_dt, end_dt):
    """
    直接在图片目录中查找风暴时段的图片
    """
    storm_images = []
    
    try:
        # 获取所有图片文件
        image_files = glob.glob(os.path.join(images_dir, "*.png"))
        print(f"Found {len(image_files)} image files in {images_dir}")
        
        # 按时间排序
        image_files.sort()
        
        # 找出风暴时段内的图片
        for img_file in image_files:
            try:
                basename = os.path.basename(img_file)
                # 处理可能的_interp后缀
                time_str = basename.split('.')[0]
                if '_interp' in time_str:
                    time_str = time_str.replace('_interp', '')
                
                img_time = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                
                if start_dt <= img_time <= end_dt:
                    storm_images.append(basename)
            except ValueError:
                continue
        
        print(f"Found {len(storm_images)} images in storm period")
        
        # 将图片分组为序列
        if len(storm_images) >= STORM_CONFIG['seq_len']:
            # 简单分组：每20个连续图片作为一个序列
            sequences = []
            for i in range(0, len(storm_images) - STORM_CONFIG['seq_len'] + 1, 5):  # 步长为5
                seq = storm_images[i:i + STORM_CONFIG['seq_len']]
                sequences.append(seq)
                print(f"Created sequence from {seq[0]} to {seq[-1]}")
            
            return sequences[:3]  # 只返回前3个序列
        
    except Exception as e:
        print(f"Error finding images directly: {e}")
    
    return []

def load_single_sequence(seq_filenames, images_dir):
    """
    加载单个序列的所有帧
    """
    frames = []
    
    for fname in seq_filenames:
        path = os.path.join(images_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not load image {fname}")
            # 创建全黑图像作为占位符
            img = np.zeros(STORM_CONFIG['img_size'], dtype=np.uint8)
        else:
            # 如果尺寸不匹配，调整尺寸
            if img.shape != STORM_CONFIG['img_size']:
                img = cv2.resize(img, STORM_CONFIG['img_size'])
        
        # 归一化到0-1
        img_normalized = img.astype(np.float32) / 255.0
        frames.append(img_normalized)
    
    # 转换为numpy数组 [20, 256, 256]
    sequence = np.array(frames)
    
    # 添加通道维度 [20, 1, 256, 256]
    sequence = np.expand_dims(sequence, axis=1)
    
    return sequence

def calculate_echo_properties(image):
    """
    计算回波属性
    image: 归一化后的图像 (0-1)
    """
    # 转换为dBZ值 (0-70 dBZ)
    dbz = image * 70.0
    
    # 1. Echo area (pixels > 15 dBZ)
    echo_mask = dbz > 15
    echo_area = np.sum(echo_mask)
    total_pixels = dbz.size
    
    # 2. Intensity metrics
    echo_pixels = dbz[echo_mask]
    if len(echo_pixels) > 0:
        mean_intensity = np.mean(echo_pixels)
        max_intensity = np.max(echo_pixels)
        intensity_std = np.std(echo_pixels)
    else:
        mean_intensity = 0
        max_intensity = 0
        intensity_std = 0
    
    # 3. Echo centroid
    if echo_area > 0:
        y_indices, x_indices = np.where(echo_mask)
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
    else:
        centroid_y, centroid_x = 0, 0
    
    # 4. Echo morphology
    if echo_area > 0 and len(y_indices) > 0 and len(x_indices) > 0:
        # Bounding box
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        bbox_area = (max_y - min_y + 1) * (max_x - min_x + 1)
        compactness = echo_area / bbox_area if bbox_area > 0 else 0
        
        # Echo shape (approximate ellipse)
        if len(y_indices) > 1:
            y_var = np.var(y_indices)
            x_var = np.var(x_indices)
            eccentricity = np.sqrt(1 - min(x_var, y_var) / max(x_var, y_var)) if max(x_var, y_var) > 0 else 0
        else:
            eccentricity = 0
    else:
        compactness = 0
        eccentricity = 0
    
    return {
        'echo_area': echo_area,
        'echo_ratio': echo_area / total_pixels if total_pixels > 0 else 0,
        'mean_intensity': mean_intensity,
        'max_intensity': max_intensity,
        'intensity_std': intensity_std,
        'centroid_y': centroid_y,
        'centroid_x': centroid_x,
        'compactness': compactness,
        'eccentricity': eccentricity
    }

def load_model(model_type, model_path):
    """
    加载训练好的模型
    """
    device = torch.device(STORM_CONFIG['device'])
    
    if model_type == 'lstm_cnn':
        model = LSTMCNN(
            input_channels=1,
            encoder_dims=[8, 16, 32],
            convlstm_dims=[32],
            decoder_dims=[32, 16, 8],
            kernel_size=(3, 3)
        )
    elif model_type == 'vit':
        model = SimpleViT(
            img_size=STORM_CONFIG['img_size'][0],
            patch_size=16,
            in_channels=1,
            embed_dim=256,
            num_heads=4,
            depth=3,
            out_channels=1,
            future_frames=STORM_CONFIG['target_len'],
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    try:
        # 加载权重
        state_dict = torch.load(model_path, map_location=device)
        
        # 尝试加载权重，使用strict=False以避免架构不匹配问题
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print(f"Successfully loaded {model_type} model")
        return model
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None

def analyze_storm_with_model(model, sequence, model_type):
    """
    用模型分析风暴序列
    """
    if model is None:
        print(f"Error: {model_type} model not available")
        return None, None, None, None
    
    device = torch.device(STORM_CONFIG['device'])
    
    # 转换为tensor [1, 20, 1, 256, 256]
    sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).float().to(device)
    
    # 分割输入和目标
    input_frames = sequence_tensor[:, :STORM_CONFIG['input_len']]
    target_frames = sequence_tensor[:, STORM_CONFIG['input_len']:]
    
    try:
        # 预测
        with torch.no_grad():
            predicted_frames = model(input_frames, STORM_CONFIG['target_len'])
        
        # 转换为numpy
        input_np = input_frames.cpu().numpy()[0, :, 0]  # [10, 256, 256]
        target_np = target_frames.cpu().numpy()[0, :, 0]  # [10, 256, 256]
        pred_np = predicted_frames.cpu().numpy()[0, :, 0]  # [10, 256, 256]
        
        # 计算属性
        analysis_results = {
            'model_type': model_type,
            'input_properties': [],
            'target_properties': [],
            'predicted_properties': [],
            'errors': {
                'movement': [],
                'intensity': [],
                'area': [],
                'mse': []
            }
        }
        
        # 分析输入帧
        for i in range(STORM_CONFIG['input_len']):
            props = calculate_echo_properties(input_np[i])
            analysis_results['input_properties'].append(props)
        
        # 分析目标帧
        for i in range(STORM_CONFIG['target_len']):
            props = calculate_echo_properties(target_np[i])
            analysis_results['target_properties'].append(props)
        
        # 分析预测帧
        for i in range(STORM_CONFIG['target_len']):
            props = calculate_echo_properties(pred_np[i])
            analysis_results['predicted_properties'].append(props)
            
            # 计算误差
            if i < len(analysis_results['target_properties']):
                target_props = analysis_results['target_properties'][i]
                
                # Movement error (centroid distance in pixels)
                movement_error = np.sqrt(
                    (props['centroid_x'] - target_props['centroid_x'])**2 +
                    (props['centroid_y'] - target_props['centroid_y'])**2
                )
                analysis_results['errors']['movement'].append(movement_error)
                
                # Intensity error
                intensity_error = abs(props['mean_intensity'] - target_props['mean_intensity'])
                analysis_results['errors']['intensity'].append(intensity_error)
                
                # Area error
                area_error = abs(props['echo_ratio'] - target_props['echo_ratio'])
                analysis_results['errors']['area'].append(area_error)
                
                # MSE per frame
                mse_error = np.mean((pred_np[i] - target_np[i])**2)
                analysis_results['errors']['mse'].append(mse_error)
        
        return analysis_results, input_np, target_np, pred_np
    
    except Exception as e:
        print(f"Error analyzing with {model_type}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def create_comparison_plots(results_lstm, results_vit, sequence_idx, output_dir):
    """
    创建LSTM-CNN和ViT模型的比较图表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查结果是否有效
    if results_lstm is None or results_vit is None:
        print(f"Skipping plots for sequence {sequence_idx} due to missing results")
        return
    
    try:
        # 1. Echo Evolution Plot
        plt.figure(figsize=(15, 10))
        
        # Extract echo ratios
        input_ratios = [p['echo_ratio'] for p in results_lstm['input_properties']]
        target_ratios = [p['echo_ratio'] for p in results_lstm['target_properties']]
        lstm_ratios = [p['echo_ratio'] for p in results_lstm['predicted_properties']]
        vit_ratios = [p['echo_ratio'] for p in results_vit['predicted_properties']]
        
        frames = list(range(1, 21))
        
        plt.subplot(2, 2, 1)
        plt.plot(frames[:10], input_ratios, 'b-o', label='Input', linewidth=2)
        plt.plot(frames[10:], target_ratios, 'g-s', label='Target', linewidth=2)
        plt.plot(frames[10:], lstm_ratios, 'r-^', label='LSTM-CNN Pred', linewidth=2)
        plt.plot(frames[10:], vit_ratios, 'm-D', label='ViT Pred', linewidth=2)
        plt.axvline(x=10.5, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Echo Coverage Ratio', fontsize=12)
        plt.title('Echo Coverage Evolution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Intensity Comparison
        plt.subplot(2, 2, 2)
        input_intensity = [p['mean_intensity'] for p in results_lstm['input_properties']]
        target_intensity = [p['mean_intensity'] for p in results_lstm['target_properties']]
        lstm_intensity = [p['mean_intensity'] for p in results_lstm['predicted_properties']]
        vit_intensity = [p['mean_intensity'] for p in results_vit['predicted_properties']]
        
        plt.plot(frames[:10], input_intensity, 'b-o', label='Input', linewidth=2)
        plt.plot(frames[10:], target_intensity, 'g-s', label='Target', linewidth=2)
        plt.plot(frames[10:], lstm_intensity, 'r-^', label='LSTM-CNN Pred', linewidth=2)
        plt.plot(frames[10:], vit_intensity, 'm-D', label='ViT Pred', linewidth=2)
        plt.axvline(x=10.5, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Mean Intensity (dBZ)', fontsize=12)
        plt.title('Echo Intensity Evolution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Error Comparison
        plt.subplot(2, 2, 3)
        error_frames = list(range(1, 11))
        
        plt.plot(error_frames, results_lstm['errors']['movement'], 'r-^', 
                 label=f'LSTM-CNN Movement Error (avg: {np.mean(results_lstm["errors"]["movement"]):.1f}px)', 
                 linewidth=2)
        plt.plot(error_frames, results_vit['errors']['movement'], 'm-D', 
                 label=f'ViT Movement Error (avg: {np.mean(results_vit["errors"]["movement"]):.1f}px)', 
                 linewidth=2)
        plt.xlabel('Prediction Frame', fontsize=12)
        plt.ylabel('Centroid Error (pixels)', fontsize=12)
        plt.title('Movement Prediction Error', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. MSE Comparison
        plt.subplot(2, 2, 4)
        plt.plot(error_frames, results_lstm['errors']['mse'], 'r-^', 
                 label=f'LSTM-CNN MSE (avg: {np.mean(results_lstm["errors"]["mse"]):.4f})', 
                 linewidth=2)
        plt.plot(error_frames, results_vit['errors']['mse'], 'm-D', 
                 label=f'ViT MSE (avg: {np.mean(results_vit["errors"]["mse"]):.4f})', 
                 linewidth=2)
        plt.xlabel('Prediction Frame', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.title('Prediction Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'storm_comparison_seq{sequence_idx}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created comparison plot: {plot_path}")
        
        # 5. Storm Track Visualization
        plt.figure(figsize=(12, 10))
        
        # Plot centroids
        input_centroids = [(p['centroid_x'], p['centroid_y']) for p in results_lstm['input_properties']]
        target_centroids = [(p['centroid_x'], p['centroid_y']) for p in results_lstm['target_properties']]
        lstm_centroids = [(p['centroid_x'], p['centroid_y']) for p in results_lstm['predicted_properties']]
        vit_centroids = [(p['centroid_x'], p['centroid_y']) for p in results_vit['predicted_properties']]
        
        # Convert to arrays
        input_arr = np.array(input_centroids)
        target_arr = np.array(target_centroids)
        lstm_arr = np.array(lstm_centroids)
        vit_arr = np.array(vit_centroids)
        
        # Plot
        plt.plot(input_arr[:, 0], input_arr[:, 1], 'b-o', label='Input Track', linewidth=2, markersize=8)
        plt.plot(target_arr[:, 0], target_arr[:, 1], 'g-s', label='True Storm Track', linewidth=2, markersize=8)
        plt.plot(lstm_arr[:, 0], lstm_arr[:, 1], 'r-^', label='LSTM-CNN Predicted Track', linewidth=2, markersize=8)
        plt.plot(vit_arr[:, 0], vit_arr[:, 1], 'm-D', label='ViT Predicted Track', linewidth=2, markersize=8)
        
        # Add arrows for direction
        for i in range(len(input_arr)-1):
            plt.arrow(input_arr[i, 0], input_arr[i, 1], 
                      input_arr[i+1, 0]-input_arr[i, 0], 
                      input_arr[i+1, 1]-input_arr[i, 1],
                      color='blue', alpha=0.5, width=0.5, head_width=3)
        
        plt.xlabel('X Position (pixels)', fontsize=12)
        plt.ylabel('Y Position (pixels)', fontsize=12)
        plt.title('Storm Track and Movement Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, STORM_CONFIG['img_size'][1])
        plt.ylim(0, STORM_CONFIG['img_size'][0])
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        
        track_path = os.path.join(output_dir, f'storm_track_seq{sequence_idx}.png')
        plt.savefig(track_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created storm track plot: {track_path}")
        
    except Exception as e:
        print(f"Error creating plots: {e}")

def create_storm_gif(input_frames, target_frames, lstm_pred, vit_pred, sequence_idx, output_dir):
    """
    创建风暴过程的GIF动画 - 修复版本
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入是否有效
    if input_frames is None or target_frames is None or lstm_pred is None or vit_pred is None:
        print(f"Skipping GIF for sequence {sequence_idx} due to missing data")
        return
    
    all_frames = []
    
    try:
        # 确定统一的图像尺寸
        standard_height, standard_width = STORM_CONFIG['img_size']
        
        # 添加输入序列的最后5帧
        for i in range(max(-5, -len(input_frames)), 0):
            frame = (input_frames[i] * 255).astype(np.uint8)
            
            # 确保帧大小一致
            if frame.shape != (standard_height, standard_width):
                frame = cv2.resize(frame, (standard_width, standard_height), interpolation=cv2.INTER_LINEAR)
            
            # 添加标签
            frame_num = len(input_frames) + i + 1
            frame = cv2.putText(
                frame.copy(), f"Input Frame {frame_num}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # 转换为RGB
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = frame
            all_frames.append(frame_rgb)
        
        # 添加预测序列和真实序列（并排显示）
        for i in range(min(len(target_frames), len(lstm_pred), len(vit_pred))):
            lstm_frame = (lstm_pred[i] * 255).astype(np.uint8)
            vit_frame = (vit_pred[i] * 255).astype(np.uint8)
            target_frame = (target_frames[i] * 255).astype(np.uint8)
            
            # 确保所有帧大小一致
            for frame in [lstm_frame, vit_frame, target_frame]:
                if frame.shape != (standard_height, standard_width):
                    frame = cv2.resize(frame, (standard_width, standard_height), interpolation=cv2.INTER_LINEAR)
            
            # 确保所有帧都是2D（灰度图）
            lstm_frame = lstm_frame if len(lstm_frame.shape) == 2 else lstm_frame[:, :, 0]
            vit_frame = vit_frame if len(vit_frame.shape) == 2 else vit_frame[:, :, 0]
            target_frame = target_frame if len(target_frame.shape) == 2 else target_frame[:, :, 0]
            
            # 添加标签
            lstm_frame = cv2.putText(
                lstm_frame.copy(), f"LSTM-CNN Pred {i+1}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            vit_frame = cv2.putText(
                vit_frame.copy(), f"ViT Pred {i+1}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            target_frame = cv2.putText(
                target_frame.copy(), f"Target {i+1}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # 转换为RGB
            lstm_rgb = cv2.cvtColor(lstm_frame, cv2.COLOR_GRAY2RGB)
            vit_rgb = cv2.cvtColor(vit_frame, cv2.COLOR_GRAY2RGB)
            target_rgb = cv2.cvtColor(target_frame, cv2.COLOR_GRAY2RGB)
            
            # 创建对比图：三列显示
            # 确保所有图像尺寸相同
            height = max(lstm_rgb.shape[0], vit_rgb.shape[0], target_rgb.shape[0])
            width = max(lstm_rgb.shape[1], vit_rgb.shape[1], target_rgb.shape[1])
            
            lstm_rgb_resized = cv2.resize(lstm_rgb, (width, height))
            vit_rgb_resized = cv2.resize(vit_rgb, (width, height))
            target_rgb_resized = cv2.resize(target_rgb, (width, height))
            
            # 创建并排对比
            top_row = np.hstack([lstm_rgb_resized, vit_rgb_resized])
            bottom_row = np.hstack([target_rgb_resized, np.zeros_like(target_rgb_resized)])  # 占位符以对齐
            
            # 如果尺寸不匹配，调整大小
            if top_row.shape[1] != bottom_row.shape[1]:
                min_width = min(top_row.shape[1], bottom_row.shape[1])
                top_row = top_row[:, :min_width, :]
                bottom_row = bottom_row[:, :min_width, :]
            
            combined = np.vstack([top_row, bottom_row])
            all_frames.append(combined)
        
        # 保存GIF
        gif_path = os.path.join(output_dir, f'storm_comparison_seq{sequence_idx}.gif')
        if all_frames:
            # 确保所有帧尺寸相同
            final_height, final_width = all_frames[0].shape[:2]
            for i in range(len(all_frames)):
                if all_frames[i].shape[:2] != (final_height, final_width):
                    all_frames[i] = cv2.resize(all_frames[i], (final_width, final_height))
            
            imageio.mimsave(gif_path, all_frames, fps=2.0, loop=0)
            print(f"Created GIF: {gif_path}")
    except Exception as e:
        print(f"Error creating GIF: {e}")
        import traceback
        traceback.print_exc()

def save_analysis_summary(all_results, output_dir):
    """
    保存分析摘要报告 - 修复Unicode编码问题
    """
    summary_file = os.path.join(output_dir, 'storm_analysis_summary.txt')
    
    try:
        # 使用UTF-8编码保存文件
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("      STRONG CONVECTION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Storm Period: {STORM_CONFIG['storm_start']} to {STORM_CONFIG['storm_end']}\n")
            f.write(f"Total Sequences Analyzed: {len(all_results)}\n\n")
            
            if len(all_results) == 0:
                f.write("No valid results to analyze.\n")
                return
            
            # 汇总所有序列的误差
            lstm_movement_errors = []
            lstm_intensity_errors = []
            lstm_mse_errors = []
            
            vit_movement_errors = []
            vit_intensity_errors = []
            vit_mse_errors = []
            
            valid_sequences = 0
            
            for seq_idx, (results_lstm, results_vit) in enumerate(all_results):
                if results_lstm is not None and results_vit is not None:
                    valid_sequences += 1
                    
                    lstm_movement_errors.extend(results_lstm['errors']['movement'])
                    lstm_intensity_errors.extend(results_lstm['errors']['intensity'])
                    lstm_mse_errors.extend(results_lstm['errors']['mse'])
                    
                    vit_movement_errors.extend(results_vit['errors']['movement'])
                    vit_intensity_errors.extend(results_vit['errors']['intensity'])
                    vit_mse_errors.extend(results_vit['errors']['mse'])
            
            f.write(f"Valid Sequences: {valid_sequences}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            if len(lstm_movement_errors) > 0:
                f.write("LSTM-CNN Model:\n")
                f.write(f"  Average Movement Error: {np.mean(lstm_movement_errors):.2f} pixels\n")
                f.write(f"  Average Intensity Error: {np.mean(lstm_intensity_errors):.2f} dBZ\n")
                f.write(f"  Average MSE: {np.mean(lstm_mse_errors):.6f}\n\n")
            else:
                f.write("LSTM-CNN Model: No valid results\n\n")
            
            if len(vit_movement_errors) > 0:
                f.write("ViT Model:\n")
                f.write(f"  Average Movement Error: {np.mean(vit_movement_errors):.2f} pixels\n")
                f.write(f"  Average Intensity Error: {np.mean(vit_intensity_errors):.2f} dBZ\n")
                f.write(f"  Average MSE: {np.mean(vit_mse_errors):.6f}\n\n")
            else:
                f.write("ViT Model: No valid results\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("=" * 60 + "\n\n")
            
            # 使用简单的ASCII字符替代特殊字符
            if len(lstm_movement_errors) > 0 and len(vit_movement_errors) > 0:
                if np.mean(lstm_movement_errors) < np.mean(vit_movement_errors):
                    f.write("* LSTM-CNN performs better in tracking storm movement\n")
                else:
                    f.write("* ViT performs better in tracking storm movement\n")
                    
                if np.mean(lstm_intensity_errors) < np.mean(vit_intensity_errors):
                    f.write("* LSTM-CNN performs better in intensity prediction\n")
                else:
                    f.write("* ViT performs better in intensity prediction\n")
                    
                if np.mean(lstm_mse_errors) < np.mean(vit_mse_errors):
                    f.write("* LSTM-CNN has better overall prediction accuracy\n")
                else:
                    f.write("* ViT has better overall prediction accuracy\n")
            else:
                f.write("* Insufficient data for comparison\n")
            
            f.write("\n* For detailed analysis, see the generated plots and GIFs\n")
        
        print(f"Analysis summary saved to: {summary_file}")
    except UnicodeEncodeError:
        # 如果UTF-8也失败，使用ASCII编码
        print("Warning: UTF-8 encoding failed, trying ASCII...")
        with open(summary_file, 'w', encoding='ascii', errors='ignore') as f:
            f.write("=" * 60 + "\n")
            f.write("      STRONG CONVECTION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Storm Period: {STORM_CONFIG['storm_start']} to {STORM_CONFIG['storm_end']}\n")
            f.write(f"Total Sequences Analyzed: {len(all_results)}\n\n")
            
            # 只写基本结果，避免特殊字符
            if len(all_results) > 0:
                f.write("Results available. See generated plots for details.\n")

# ================= 主函数 =================
def main():
    print("=" * 60)
    print("STRONG CONVECTION STORM ANALYSIS")
    print(f"Date: {STORM_CONFIG['storm_start'][:8]}")
    print("=" * 60 + "\n")
    
    # 1. 查找项目文件
    print("Step 1: Searching for project files...")
    selected_files = find_project_files()
    
    # 2. 检查文件是否存在
    print("\nStep 2: Checking if selected files exist...")
    if not check_files_exist(selected_files):
        print("\nSome required files are missing!")
        print("\nPlease manually check:")
        print("1. Ensure processed_data directory exists with train_list.txt")
        print("2. Ensure checkpoints directory exists with model files")
        print("3. Ensure images are in processed_data/images/")
        return
    
    # 创建输出目录
    output_dir = STORM_CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 查找风暴序列
    print("\nStep 3: Finding storm sequences...")
    
    if 'list_file' not in selected_files or 'images_dir' not in selected_files:
        print("Missing required files for sequence loading!")
        return
    
    storm_indices, storm_sequences = load_storm_sequences(
        selected_files['images_dir'], 
        selected_files['list_file']
    )
    
    if not storm_sequences:
        print("\nNo storm sequences found!")
        return
    
    # 4. 加载模型
    print("\nStep 4: Loading models...")
    
    model_paths = selected_files.get('model_paths', {})
    
    lstm_model = None
    if 'lstm_cnn' in model_paths:
        lstm_model = load_model('lstm_cnn', model_paths['lstm_cnn'])
    
    vit_model = None
    if 'vit' in model_paths:
        vit_model = load_model('vit', model_paths['vit'])
    
    if lstm_model is None and vit_model is None:
        print("\nWarning: No models could be loaded!")
        print("Analysis will continue without model predictions")
    
    # 5. 分析每个风暴序列（只分析前5个以节省时间）
    print(f"\nStep 5: Analyzing {min(5, len(storm_sequences))} storm sequences (out of {len(storm_sequences)} found)...")
    all_results = []
    
    # 只分析前5个序列
    sequences_to_analyze = storm_sequences[:5]
    
    for seq_idx, seq_filenames in enumerate(sequences_to_analyze):
        print(f"\nAnalyzing sequence {seq_idx + 1}/{len(sequences_to_analyze)}...")
        print(f"Sequence files: {seq_filenames[0]} ... {seq_filenames[-1]}")
        
        # 加载序列数据
        sequence_data = load_single_sequence(seq_filenames, selected_files['images_dir'])
        
        # 用LSTM-CNN分析（如果模型可用）
        results_lstm, input_frames, target_frames, lstm_pred = None, None, None, None
        if lstm_model is not None:
            results_lstm, input_frames, target_frames, lstm_pred = analyze_storm_with_model(
                lstm_model, sequence_data, 'lstm_cnn'
            )
        
        # 用ViT分析（如果模型可用）
        results_vit, _, _, vit_pred = None, None, None, None
        if vit_model is not None:
            results_vit, _, _, vit_pred = analyze_storm_with_model(
                vit_model, sequence_data, 'vit'
            )
        
        # 保存结果
        all_results.append((results_lstm, results_vit))
        
        # 创建可视化（如果两个模型的结果都可用）
        if results_lstm is not None and results_vit is not None:
            create_comparison_plots(results_lstm, results_vit, seq_idx, output_dir)
            
            # 创建GIF
            create_storm_gif(input_frames, target_frames, lstm_pred, vit_pred, 
                            seq_idx, output_dir)
        elif results_lstm is not None or results_vit is not None:
            print(f"  Only one model available, skipping comparison plots")
        else:
            print(f"  No model results available")
        
        print(f"  Sequence {seq_idx} analysis completed")
    
    # 6. 保存分析摘要
    print("\nStep 6: Generating analysis summary...")
    save_analysis_summary(all_results, output_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Results saved in: {output_dir}")
    
    # 显示生成的文件
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        if files:
            print(f"\nGenerated {len(files)} files:")
            for file in sorted(files):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path) / 1024  # KB
                ext = os.path.splitext(file)[1]
                if ext in ['.png', '.gif']:
                    print(f"  - {file} ({size:.1f} KB)")
                elif ext == '.txt':
                    print(f"  - {file} (analysis summary)")
    
    print("\nKey analysis results:")
    print("1. Comparison plots show echo evolution, intensity, and errors")
    print("2. Storm track plots show movement prediction accuracy")
    print("3. GIFs show visual comparison between models")
    print("4. Summary file contains quantitative analysis")
    print("=" * 60)

if __name__ == "__main__":
    main()