import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq
from scipy import signal # 确保导入 signal

# --- 0. 配置参数 ---
TARGET_PRED_LEN = 96
TARGET_DATASETS = ["ECL", "Traffic", "Weather"]
TARGET_MODELS = ["iTransformer", "RLinear", "PatchTST", "ModernTCN", "FITS", "DLinear", "GPT4TS"]
SRRF_SUFFIX = "+SRRF" # 用于图例中标识SRRF增强版

NUM_BANDS = 5
SAMPLING_RATE = 1.0 # 假设单位为 样本/小时，则频率单位为 周期/小时
FREQ_UNIT_LABEL_SHORT = "cyc/hr" # 用于频带名称

# 绘图颜色配置
COLOR_NON_RAG = '#B0C4DE'       # LightSteelBlue - 基础模型
COLOR_DELTA_POSITIVE = '#2E8B57' # SeaGreen - 正向增量/改进
COLOR_DELTA_NEGATIVE = '#CD5C5C' # IndianRed - 负向增量/恶化

# --- Helper Functions (FFT, PSD, Metrics) ---
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建目录: {directory}")

def compute_fft_magnitude(data_series):
    if data_series is None or len(data_series) == 0:
        return np.array([])
    windowed_data = data_series * np.hanning(len(data_series))
    fft_result = fft(windowed_data)
    n = len(data_series)
    magnitude = np.abs(fft_result[:n // 2]) / n
    return magnitude

def calculate_freq_rmse(true_spectrum_mag, pred_spectrum_mag):
    min_len = min(len(true_spectrum_mag), len(pred_spectrum_mag))
    if min_len == 0:
        return float('nan')
    true_s = true_spectrum_mag[:min_len]
    pred_s = pred_spectrum_mag[:min_len]
    return np.sqrt(np.mean((true_s - pred_s) ** 2))

def get_band_energy(psd_values, frequencies, band_start_freq, band_end_freq):
    if len(frequencies) == 0 or len(psd_values) == 0:
        return 0.0
    band_indices = np.logical_and(frequencies >= band_start_freq, frequencies < band_end_freq)
    return np.sum(psd_values[band_indices]) if np.any(band_indices) else 0.0

def calculate_all_band_energies_for_sample(data_series, sampling_rate, num_bands):
    if data_series is None or len(data_series) == 0:
        return [0.0] * num_bands
        
    frequencies, psd_values = signal.welch(data_series, fs=sampling_rate, nperseg=min(256, len(data_series)))
    if len(frequencies) == 0: # welch 可能返回空
        return [0.0] * num_bands

    nyquist_freq = sampling_rate / 2
    band_edges = np.linspace(0, nyquist_freq, num_bands + 1)
    energies = []
    for i in range(num_bands):
        energies.append(get_band_energy(psd_values, frequencies, band_edges[i], band_edges[i+1]))
    return energies

# --- 1. 数据加载和处理 ---
def process_all_experiment_data(parquet_path="result_summary.parquet"):
    print("开始处理实验数据...")
    try:
        summary_df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"错误: '{parquet_path}' 未找到。请确保文件路径正确。")
        return None, None

    # 筛选出每个 (dataset, pred_len, model_name) 组合下 mse 最优的配置
    best_config_rows = summary_df.loc[
        summary_df.groupby(['dataset', 'pred_len', 'model_name'])['mse'].idxmin()
    ].reset_index(drop=True)

    # 初始化存储计算结果的字典
    # Freq_RMSE: rmse_data[dataset][model] = {"non_rag": val, "rag": val}
    # Band Energies: band_energy_data[dataset][model][band_idx] = {"non_rag": val, "rag": val}
    rmse_data = {ds: {model: {} for model in TARGET_MODELS} for ds in TARGET_DATASETS}
    band_energy_data = {ds: {model: {b_idx: {} for b_idx in range(NUM_BANDS)} for model in TARGET_MODELS} for ds in TARGET_DATASETS}

    processed_configs = 0
    for _, row in best_config_rows.iterrows():
        dataset = row["dataset"]
        pred_len = row["pred_len"]
        base_model_name = row["model_name"] # 这是基础模型的名称

        if dataset not in TARGET_DATASETS or base_model_name not in TARGET_MODELS or pred_len != TARGET_PRED_LEN:
            continue
        
        print(f"\n处理: Dataset={dataset}, Model={base_model_name}, PredLen={pred_len}")

        # 构建路径 (假设 raw_name 对应基础模型的实验ID)
        # raw_name 示例: ETTh1_96_96_iTransformer_ETTh1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0
        base_exp_id = row["raw_name"]
        if base_exp_id.endswith("_rag"): # 如果raw_name本身是RAG的，去掉后缀得到base
            base_exp_id = base_exp_id[:-4]
            
        base_model_folder = os.path.join("results", "long_term_forecast_" + base_exp_id)
        rag_model_folder = base_model_folder + "_rag"

        true_path = os.path.join(base_model_folder, "true.npy")
        base_pred_path = os.path.join(base_model_folder, "pred.npy")
        rag_pred_path = os.path.join(rag_model_folder, "pred.npy")

        try:
            true_values = np.load(true_path)      # Shape: (num_samples, seq_len, num_features)
            base_pred_values = np.load(base_pred_path)
            rag_pred_values = np.load(rag_pred_path)
            print(f"  成功加载 .npy 文件。真实值shape: {true_values.shape}")
        except FileNotFoundError as e:
            print(f"  错误: 找不到 .npy 文件 for {base_model_name} in {dataset}: {e}")
            continue
        
        num_exp_samples = true_values.shape[0]
        num_exp_features = true_values.shape[2]

        # --- 计算 Freq_RMSE ---
        all_rmse_base, all_rmse_rag = [], []
        for s_idx in range(num_exp_samples):
            for f_idx in range(num_exp_features):
                true_mag = compute_fft_magnitude(true_values[s_idx, :, f_idx])
                base_mag = compute_fft_magnitude(base_pred_values[s_idx, :, f_idx])
                rag_mag = compute_fft_magnitude(rag_pred_values[s_idx, :, f_idx])

                if len(true_mag) > 0:
                    if len(base_mag) > 0:
                        all_rmse_base.append(calculate_freq_rmse(true_mag, base_mag))
                    if len(rag_mag) > 0:
                        all_rmse_rag.append(calculate_freq_rmse(true_mag, rag_mag))
        
        avg_rmse_base = np.nanmean(all_rmse_base) if all_rmse_base else float('nan')
        avg_rmse_rag = np.nanmean(all_rmse_rag) if all_rmse_rag else float('nan')
        rmse_data[dataset][base_model_name] = {"non_rag": avg_rmse_base, "rag": avg_rmse_rag}
        print(f"  Avg Freq_RMSE: Base={avg_rmse_base:.4f}, RAG={avg_rmse_rag:.4f}")

        # --- 计算频带能量 ---
        sample_band_energies_true = [[] for _ in range(NUM_BANDS)]
        sample_band_energies_base = [[] for _ in range(NUM_BANDS)]
        sample_band_energies_rag  = [[] for _ in range(NUM_BANDS)]

        for s_idx in range(num_exp_samples):
            for f_idx in range(num_exp_features):
                true_energies = calculate_all_band_energies_for_sample(true_values[s_idx, :, f_idx], SAMPLING_RATE, NUM_BANDS)
                base_energies = calculate_all_band_energies_for_sample(base_pred_values[s_idx, :, f_idx], SAMPLING_RATE, NUM_BANDS)
                rag_energies  = calculate_all_band_energies_for_sample(rag_pred_values[s_idx, :, f_idx], SAMPLING_RATE, NUM_BANDS)
                for b_idx in range(NUM_BANDS):
                    sample_band_energies_true[b_idx].append(true_energies[b_idx])
                    sample_band_energies_base[b_idx].append(base_energies[b_idx])
                    sample_band_energies_rag[b_idx].append(rag_energies[b_idx])
        
        for b_idx in range(NUM_BANDS):
            avg_band_energy_true = np.nanmean(sample_band_energies_true[b_idx]) if sample_band_energies_true[b_idx] else float('nan') # Not used directly in plot but good to have
            avg_band_energy_base = np.nanmean(sample_band_energies_base[b_idx]) if sample_band_energies_base[b_idx] else float('nan')
            avg_band_energy_rag  = np.nanmean(sample_band_energies_rag[b_idx]) if sample_band_energies_rag[b_idx] else float('nan')
            band_energy_data[dataset][base_model_name][b_idx] = {"non_rag": avg_band_energy_base, "rag": avg_band_energy_rag}
            # print(f"  Avg Band {b_idx+1} Energy: Base={avg_band_energy_base:.4e}, RAG={avg_band_energy_rag:.4e}")
        processed_configs +=1

    print(f"\n数据处理完成。共处理了 {processed_configs} 个有效配置。")
    return rmse_data, band_energy_data


# --- 2. 绘图函数 (与之前版本类似，但调整了样式和调用方式) ---
def plot_rmse_comparison_actual_data(rmse_plot_data, save_dir):
    """为Freq_RMSE绘制包含三个子图的大图 (使用真实数据)"""
    metric_display_name = "Freq_RMSE"
    fig, axes = plt.subplots(1, len(TARGET_DATASETS), figsize=(18, 6.5), sharey=True)
    if len(TARGET_DATASETS) == 1: axes = [axes]

    bar_width = 0.6
    
    for i, dataset_name in enumerate(TARGET_DATASETS):
        ax = axes[i]
        dataset_data = rmse_plot_data.get(dataset_name, {}) # 使用 .get 以防数据集数据缺失
        
        model_names_on_xaxis = TARGET_MODELS # 确保按固定顺序绘制
        x_positions = np.arange(len(model_names_on_xaxis))

        non_rag_values = [dataset_data.get(model, {}).get("non_rag", np.nan) for model in model_names_on_xaxis]
        rag_values = [dataset_data.get(model, {}).get("rag", np.nan) for model in model_names_on_xaxis]
        deltas = [r - nr if pd.notna(r) and pd.notna(nr) else np.nan for r, nr in zip(rag_values, non_rag_values)]

        ax.bar(x_positions, non_rag_values, bar_width, color=COLOR_NON_RAG, zorder=2, label="Base Model" if i==0 else "")

        for j, delta in enumerate(deltas):
            if pd.isna(delta) or pd.isna(non_rag_values[j]): continue
            current_color = COLOR_DELTA_POSITIVE if delta <= 0 else COLOR_DELTA_NEGATIVE # RMSE越小越好，所以正delta是恶化
            ax.bar(x_positions[j], delta, bar_width, bottom=non_rag_values[j],
                   color=current_color, alpha=0.85, zorder=3, 
                   label="Delta by +SRRF (RMSE Lower is Better)" if i==0 and j==0 and delta <=0 else ("Delta by +SRRF (RMSE Higher is Worse)" if i==0 and j==0 and delta > 0 else ""))
            
            if abs(delta) > 1e-5: 
                annotation_y_pos_delta = non_rag_values[j] + delta / 2
                min_delta_height_for_internal_text = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                if abs(delta) < min_delta_height_for_internal_text:
                    va_align_delta = 'bottom' if delta >= 0 else 'top'
                    offset_delta = 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    annotation_y_pos_delta = non_rag_values[j] + delta + (offset_delta if delta >=0 else -offset_delta*1.5)
                else:
                    va_align_delta = 'center'
                ax.text(x_positions[j], annotation_y_pos_delta, f'{delta:+.4f}', 
                        ha='center', va=va_align_delta, fontsize=7, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.3, ec='none') if abs(delta) < min_delta_height_for_internal_text else None,
                        zorder=4)

        ax.set_ylabel(metric_display_name if i == 0 else "", fontsize=11)
        ax.set_title(dataset_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_names_on_xaxis, rotation=45, ha="right", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6, axis='y', zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 调整图例标签以反映RMSE的意义
    handles = [plt.Rectangle((0,0),1,1,color=COLOR_NON_RAG),
               plt.Rectangle((0,0),1,1,color=COLOR_DELTA_POSITIVE, alpha=0.85), # Positive delta in RMSE is worse if base is positive
               plt.Rectangle((0,0),1,1,color=COLOR_DELTA_NEGATIVE, alpha=0.85)] # Negative delta in RMSE is better
    labels = ['Base Model Freq_RMSE', '+SRRF: Freq_RMSE Change (Worse if >0)', '+SRRF: Freq_RMSE Change (Better if <0)']


    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=False)
    # plt.tight_layout(rect=[0, 0.15, 1, 0.98]) # 增加底部空间给图例
    
    filename = os.path.join(save_dir, "Freq_RMSE_Comparison.pdf") # 保存为PDF
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Freq_RMSE对比图已保存: {filename}")


def plot_dataset_all_bands_energy_actual_data(dataset_name, energy_data_for_this_dataset, save_dir):
    """为单个数据集绘制包含所有频段能量对比的宽图 (使用真实数据)"""
    fig, ax = plt.subplots(figsize=(18, 7.5))
    # ax.set_title(f"{dataset_name}: Average Frequency Band Energy Comparison", fontsize=16, fontweight='bold', pad=20)

    n_models = len(TARGET_MODELS)
    n_bands = NUM_BANDS
    
    bar_group_width = 0.8 
    bar_width = bar_group_width / n_bands 
    x_model_positions = np.arange(n_models)

    for model_idx, model_name in enumerate(TARGET_MODELS):
        model_band_data = energy_data_for_this_dataset.get(model_name, {})
        
        non_rag_band_values = [model_band_data.get(b_idx, {}).get("non_rag", np.nan) for b_idx in range(n_bands)]
        rag_band_values = [model_band_data.get(b_idx, {}).get("rag", np.nan) for b_idx in range(n_bands)]
        deltas_band = [r - nr if pd.notna(r) and pd.notna(nr) else np.nan for r, nr in zip(rag_band_values, non_rag_band_values)]

        band_offsets = np.arange(n_bands) * bar_width - bar_group_width/2 + bar_width/2
        current_x_band_positions = x_model_positions[model_idx] + band_offsets

        ax.bar(current_x_band_positions, non_rag_band_values, bar_width * 0.9, 
               color=COLOR_NON_RAG, zorder=2, 
               label='Base Model Energy' if model_idx == 0 else "")

        for band_idx, delta in enumerate(deltas_band):
            if pd.isna(delta) or pd.isna(non_rag_band_values[band_idx]): continue
            current_color = COLOR_DELTA_POSITIVE if delta >= 0 else COLOR_DELTA_NEGATIVE
            ax.bar(current_x_band_positions[band_idx], delta, bar_width * 0.9,
                   bottom=non_rag_band_values[band_idx],
                   color=current_color, alpha=0.85, zorder=3,
                   label=f'Delta by {SRRF_SUFFIX} (Positive)' if model_idx == 0 and band_idx == 0 and delta >=0 else (f'Delta by {SRRF_SUFFIX} (Negative)' if model_idx == 0 and band_idx == 0 and delta <0 else "") )
            
            if abs(delta) > 1e-5:
                annotation_y_pos_delta = non_rag_band_values[band_idx] + delta / 2
                min_delta_height_for_internal_text = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                if abs(delta) < min_delta_height_for_internal_text :
                    va_align_delta = 'bottom' if delta >= 0 else 'top'
                    offset_delta = 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    annotation_y_pos_delta = non_rag_band_values[band_idx] + delta + (offset_delta if delta >=0 else -offset_delta*1.5)
                else:
                    va_align_delta = 'center'
                ax.text(current_x_band_positions[band_idx], annotation_y_pos_delta, f'{delta:+.4f}', 
                        ha='center', va=va_align_delta, fontsize=6, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.3, ec='none') if abs(delta) < min_delta_height_for_internal_text else None,
                        zorder=4)

    ax.set_ylabel("Average Energy Value", fontsize=11)
    ax.set_xlabel("Models", fontsize=11) # 主X轴标签是模型
    ax.set_xticks(x_model_positions)
    ax.set_xticklabels(TARGET_MODELS, ha="center", fontsize=10)

    ax.grid(True, linestyle='--', alpha=0.6, axis='y', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    handles, labels = ax.get_legend_handles_labels() # 获取当前子图的图例项
    # 去重图例标签
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)

    # plt.tight_layout(rect=[0, 0.12, 1, 0.93]) 
    filename = os.path.join(save_dir, f"Energy_AllBands_{dataset_name.replace(' ', '_')}.pdf") # 保存为PDF
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"频带能量对比图已保存: {filename}")


# --- 3. 主执行逻辑 ---
def main_actual_data():
    output_directory = "freq_metric"
    ensure_dir(output_directory)

    rmse_results, band_energy_results = process_all_experiment_data()

    if rmse_results is None or band_energy_results is None:
        print("数据处理失败，无法生成图表。")
        return

    # 绘制 Freq_RMSE 对比图
    print("\n开始绘制 Freq_RMSE 对比图...")
    plot_rmse_comparison_actual_data(rmse_results, output_directory)

    # 绘制每个数据集的频带能量对比图
    print("\n开始绘制频带能量对比图...")
    for dataset_name in TARGET_DATASETS:
        if dataset_name in band_energy_results:
            print(f"  为数据集 '{dataset_name}' 绘制频带能量图...")
            plot_dataset_all_bands_energy_actual_data(
                dataset_name,
                band_energy_results[dataset_name],
                output_directory
            )
        else:
            print(f"  警告: 数据集 '{dataset_name}' 的频带能量数据未找到，跳过绘图。")
    
    print(f"\n所有图表已生成并保存在 '{output_directory}' 目录下。")

if __name__ == "__main__":
    main_actual_data()
