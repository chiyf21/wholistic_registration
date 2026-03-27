import os
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import sys

# Add the src directory to the path using absolute path
sys.path.append(os.path.abspath('./code/wbi_0123/wholistic_registration/src/wholistic_registration'))

# Import modules
from utils import preprocess as prep
from utils import option
from utils import calFlowCrossResolution, calFlow3d_Wei_v1

# -----------------------------
# 基本配置
# -----------------------------

root_dir = Path('/home/cyf/wbi/Virginia/code/src/wholistic_registration/exp/simulation/repos')
zslices = [2, 10, 18]  # 选择 z slice
output_file = root_dir / 'MSE_summary.npy'

# -----------------------------
# 遍历实验组
# -----------------------------
experiment_groups = ['art_R', 'amp', 'noise_level']  # 对应三组实验
results = {}  # 保存 MSE

# 初始化结果存储结构
# results[group][exp_level][method][repo_idx] = mse_value
results = {group: {} for group in experiment_groups}

# 遍历所有repo目录
repo_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith('rep')])
num_repos = len(repo_dirs)
print(f"Found {num_repos} repos: {[d.name for d in repo_dirs]}")

for repo_idx, repo_dir in enumerate(repo_dirs):
    print(f"\n=== Processing Repo {repo_dir.name} ({repo_idx+1}/{num_repos}) ===")
    
    for group in experiment_groups:
        group_dir = repo_dir / group
        if not group_dir.exists():
            print(f"Warning: {group_dir} not exist, skipping")
            continue

        # 遍历每个子实验文件夹
        for subfolder in sorted(group_dir.iterdir(), key=lambda x: int(x.name)):
            if not subfolder.is_dir():
                continue
            
            exp_level = subfolder.name
            print(f"  Processing {group}/{exp_level}")

            # -----------------------------
            # 读取数据
            # -----------------------------
            ref_path = subfolder / 'ref.tif'
            mov_path = subfolder / 'mov.tif'  # raw_mov 文件名固定
            
            ref_data = tifffile.imread(ref_path)[..., 2:19].transpose(1,0,2)
            move_data = tifffile.imread(mov_path)[..., zslices].transpose(1,0,2)
            ref_data_clip = ref_data[:,:,[0,8,16]]
            
            # -----------------------------
            # option 配置（固定）
            # -----------------------------
            option['zRatio'] = 3.0769230769230766 * 8
            option['zRatio_HR'] = 3.0769230769230766
            option['layer'] = 1
            option['iter'] = 10
            option['smoothPenalty_raw'] = 0.03
            option['mask_ref'] = np.zeros(ref_data.shape, dtype=np.bool_)
            option['mask_mov'] = np.zeros(move_data.shape, dtype=np.bool_)
            option['motion'] = np.zeros([move_data.shape[0],move_data.shape[1],move_data.shape[2],3], dtype=np.bool_)
            Pnltfactor = prep.getSmPnltNormFctr(ref_data, option)
            smoothPenalty = Pnltfactor * option['smoothPenalty_raw']
            option['smoothPenalty'] = smoothPenalty
            option['movRange'] = 10

            # -----------------------------
            # 运行 warp 方法
            # -----------------------------
            motion_warp, _, _, _ = calFlow3d_Wei_v1.getMotion(ref_data_clip, move_data, option, verbose=False)
            data_mov_corrected = calFlow3d_Wei_v1.correctMotion(move_data, motion_warp)
            mse_warp = np.mean((data_mov_corrected - move_data)**2)

            # -----------------------------
            # 运行 map 方法
            # -----------------------------
            phase_map, motion_map, data_mov_mapped = calFlowCrossResolution.getMotion_v2(move_data, ref_data, option, verbose=False)
            mse_map = np.mean((data_mov_mapped.get() - move_data)**2)

            print(f"    MSE warp: {mse_warp:.4f}, MSE map: {mse_map:.4f}")

            # 存储结果
            if exp_level not in results[group]:
                results[group][exp_level] = {
                    'mse_warp': [],
                    'mse_map': []
                }
            results[group][exp_level]['mse_warp'].append(mse_warp)
            results[group][exp_level]['mse_map'].append(mse_map)

# 计算平均值和标准差
processed_results = {}
for group in experiment_groups:
    if group not in results or not results[group]:
        continue
    
    # 按实验级别排序
    sorted_exp_levels = sorted(results[group].keys(),key=int)
    
    # 计算每个实验级别的平均值和标准差
    avg_mse_warp = []
    std_mse_warp = []
    avg_mse_map = []
    std_mse_map = []
    
    for exp_level in sorted_exp_levels:
        # warp方法
        warp_values = results[group][exp_level]['mse_warp']
        avg_mse_warp.append(np.mean(warp_values))
        std_mse_warp.append(np.std(warp_values))
        
        # map方法
        map_values = results[group][exp_level]['mse_map']
        avg_mse_map.append(np.mean(map_values))
        std_mse_map.append(np.std(map_values))
    
    processed_results[group] = {
        'labels': sorted_exp_levels,
        'avg_mse_warp': avg_mse_warp,
        'std_mse_warp': std_mse_warp,
        'avg_mse_map': avg_mse_map,
        'std_mse_map': std_mse_map
    }

# 保存 MSE 数据
output_dir = output_file.parent
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_file, processed_results)
print(f"Saved processed MSE summary to {output_file}")

# -----------------------------# 绘制曲线图# -----------------------------# 创建包含3个子图的大图（1行3列）
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# 为每种方法分配固定的颜色
method_colors = {
    'Warping': 'blue',  # warp方法用蓝色
    'Mapping': 'red'    # map方法用红色
}

# 子图标题映射
group_titles = {
    'art_R': 'Artificial Resolution',
    'amp': 'Amplitude Variation',
    'noise_level': 'Noise Level'
}

# 在每个子图中绘制对应实验组的结果
for idx, group in enumerate(experiment_groups):
    if group not in processed_results:
        continue  # 如果该组没有结果，跳过
        
    ax = axes[idx]
    labels = processed_results[group]['labels']
    x = np.arange(len(labels))
    
    # warp方法 - 绘制平均曲线和误差带
    avg_warp = processed_results[group]['avg_mse_warp']
    std_warp = processed_results[group]['std_mse_warp']
    ax.plot(x, avg_warp, '-o', color=method_colors['Warping'], label='warp Method')
    ax.fill_between(x, np.array(avg_warp) - np.array(std_warp), np.array(avg_warp) + np.array(std_warp), 
                    color=method_colors['Warping'], alpha=0.2, label='Warping Error Band')
    
    # map方法 - 绘制平均曲线和误差带
    avg_map = processed_results[group]['avg_mse_map']
    std_map = processed_results[group]['std_mse_map']
    ax.plot(x, avg_map, '--s', color=method_colors['Mapping'], label='Mapping Method')
    ax.fill_between(x, np.array(avg_map) - np.array(std_map), np.array(avg_map) + np.array(std_map), 
                    color=method_colors['Mapping'], alpha=0.2, label='Mapping Error Band')
    
    # 设置子图标题和标签
    ax.set_title(group_titles[group])
    ax.set_xlabel('Experiment Level')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()

# 设置共用的y轴标签
axes[0].set_ylabel('MSE (Average ± Std)')

# 设置大图标题
fig.suptitle('Comparison of warp vs map Methods Across 3 Experiment Groups (10 Repos Average)', fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.85)  # 调整顶部空间以容纳大图标题
plt.show()

# 保存绘制的图形
graph_path = output_dir / 'mse_comparison_avg.png'
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
print(f"Graph saved to {graph_path}")
