import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import os


def load_sample_data(path):
    try:
        true_values = np.load(path + "/true.npy")
        model1_pred = np.load(path + "/pred.npy")
        model2_pred = np.load(path + "_rag" + "/pred.npy")
        print(
            f"数据加载成功。形状: 真实值={true_values.shape}, 模型1={model1_pred.shape}, 模型2={model2_pred.shape}"
        )
        return true_values, model1_pred, model2_pred
    except FileNotFoundError as e:
        print(f"错误：找不到数据文件。请检查路径 '{path}' 是否正确。")
        print(e)


def ensure_dir(directory):
    """如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建目录: {directory}")
    return directory


def compute_fft(data):
    """计算时间序列数据的傅里叶变换"""
    if data is None or len(data) == 0:
        return np.array([]), np.array([])
    windowed_data = data * np.hanning(len(data))
    fft_result = fft(windowed_data)
    n = len(data)
    fft_result = fft_result[: n // 2]
    magnitude = np.abs(fft_result) / n
    freq = fftfreq(n, 1)[: n // 2]  # d=1, 单位是采样间隔
    return freq, magnitude


def plot_frequency_spectrum(
    model_input,  # 模型的输入数据 (sequence_length,)
    true_data_sample,  # 单个样本 (sequence_length,)
    model1_data_sample,  # 单个样本 (sequence_length,)
    model2_data_sample,  # 单个样本 (sequence_length,)
    sample_id_for_title,
    feature_id_for_title,
    model1_name="iTransformer",
    model2_name="+SRRF",
    sampling_rate=1.0,
    freq_unit_label="Frequency (cycles/hour)",
    save_dir="freq_results",
    filename_suffix="",
):
    """绘制单个样本的频谱比较图"""
    freq_true, mag_true = compute_fft(true_data_sample)
    freq_model1, mag_model1 = compute_fft(model1_data_sample)
    freq_model2, mag_model2 = compute_fft(model2_data_sample)

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(
        np.concatenate([model_input, true_data_sample]),
        # "k",
        label="Ground Truth",
        linewidth=2,
    )

    plt.plot(
        np.concatenate([model_input, model1_data_sample]),
        # "b",
        label=model1_name,
        linewidth=2,
    )
    plt.plot(
        np.concatenate([model_input, model2_data_sample]),
        # "r",
        label=model2_name,
        linewidth=2,
    )

    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel(f"MSE", fontsize=12)
    # plt.title(
    #     f"Time Domain Signal - Model2 Best Case (Sample {sample_id_for_title}, Feature {feature_id_for_title})",
    #     fontsize=14,
    # )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    plt.subplot(1, 2, 2)
    if len(freq_true) > 0:
        plt.semilogy(
            freq_true * sampling_rate, mag_true, "k-", label="Ground Truth", linewidth=2
        )
    if len(freq_model1) > 0:
        plt.semilogy(
            freq_model1 * sampling_rate,
            mag_model1,
            "b--",
            label=model1_name,
            linewidth=2,
        )
    if len(freq_model2) > 0:
        plt.semilogy(
            freq_model2 * sampling_rate,
            mag_model2,
            "r-.",
            label=model2_name,
            linewidth=2,
        )

    plt.xlabel(freq_unit_label, fontsize=12)
    plt.ylabel("Magnitude (log scale)", fontsize=12)
    # plt.title(
    #     f"Frequency Spectrum - Model2 Best Case (Sample {sample_id_for_title}, Feature {feature_id_for_title})",
    #     fontsize=14,
    # )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    if sampling_rate > 0 and len(freq_true) > 0:
        # 使用 compute_fft 返回的实际最大频率乘以 sampling_rate 作为上限
        max_display_freq = freq_true.max() * sampling_rate
        plt.xlim(
            0, max_display_freq * 1.05 if max_display_freq > 0 else sampling_rate / 2
        )
    elif sampling_rate > 0:
        plt.xlim(0, sampling_rate / 2)
    else:
        plt.xlim(0)

    plt.tight_layout()
    save_filename = f"{save_dir}/frequency_spectrum_sample{sample_id_for_title}_feature{feature_id_for_title}{filename_suffix}.pdf"
    plt.savefig(save_filename, format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.1)
    plt.close()
    print(f"  频谱图已保存: {save_filename}")


def compute_power_spectral_density(data, fs=1.0):
    """计算功率谱密度"""
    if (
        data is None or len(data) < min(256, len(data)) or len(data) == 0
    ):  # 确保数据长度足够进行Welch
        return np.array([]), np.array([])
    # nperseg 不能大于 len(data)
    nperseg_val = min(256, len(data))
    if nperseg_val == 0:
        return np.array([]), np.array([])

    f, Pxx = signal.welch(data, fs=fs, nperseg=nperseg_val)
    return f, Pxx


def plot_power_spectral_density(
    true_data_sample,  # 单个样本 (sequence_length,)
    model1_data_sample,  # 单个样本 (sequence_length,)
    model2_data_sample,  # 单个样本 (sequence_length,)
    sample_id_for_title,
    feature_id_for_title,
    model1_name="iTransformer",
    model2_name="+SRRF",
    sampling_rate=1.0,
    freq_unit_label="Frequency (cycles/hour)",
    save_dir="freq_results",
    filename_suffix="",
):
    """绘制单个样本的功率谱密度比较图"""
    f_true, Pxx_true = compute_power_spectral_density(
        true_data_sample, fs=sampling_rate
    )
    f_model1, Pxx_model1 = compute_power_spectral_density(
        model1_data_sample, fs=sampling_rate
    )
    f_model2, Pxx_model2 = compute_power_spectral_density(
        model2_data_sample, fs=sampling_rate
    )

    plt.figure(figsize=(12, 6))
    if len(f_true) > 0:
        plt.semilogy(f_true, Pxx_true, "k-", label="Ground Truth", linewidth=2)
    if len(f_model1) > 0:
        plt.semilogy(f_model1, Pxx_model1, "b--", label=model1_name, linewidth=2)
    if len(f_model2) > 0:
        plt.semilogy(f_model2, Pxx_model2, "r-.", label=model2_name, linewidth=2)

    plt.xlabel(freq_unit_label, fontsize=12)
    plt.ylabel("PSD (log scale)", fontsize=12)
    plt.title(
        f"Power Spectral Density - Model2 Best Case (Sample {sample_id_for_title}, Feature {feature_id_for_title})",
        fontsize=14,
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    if sampling_rate > 0:
        plt.xlim(0, sampling_rate / 2)
    else:
        plt.xlim(0)

    plt.tight_layout()
    save_filename = f"{save_dir}/psd_sample{sample_id_for_title}_feature{feature_id_for_title}{filename_suffix}.png"
    plt.savefig(save_filename, dpi=300)
    plt.close()
    print(f"  PSD图已保存: {save_filename}")


def calculate_frequency_domain_metrics(true_spectrum, pred_spectrum):
    """计算频域评估指标"""
    min_len = min(len(true_spectrum), len(pred_spectrum))
    if min_len == 0:
        return {
            "Freq_MAE": float("nan"),
            "Freq_MSE": float("nan"),
            "Freq_RMSE": float("nan"),
            "Freq_Correlation": float("nan"),
            "Energy_Ratio": float("nan"),
        }
    true_spectrum = true_spectrum[:min_len]
    pred_spectrum = pred_spectrum[:min_len]

    freq_mae = np.mean(np.abs(pred_spectrum - true_spectrum))
    freq_mse = np.mean((pred_spectrum - true_spectrum) ** 2)
    freq_rmse = np.sqrt(freq_mse)

    if (
        np.std(true_spectrum) < 1e-9 or np.std(pred_spectrum) < 1e-9 or min_len < 2
    ):  # 增加min_len < 2的判断
        corr = float("nan")
    else:
        corr_matrix = np.corrcoef(true_spectrum, pred_spectrum)
        corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else float("nan")

    energy_true = np.sum(np.abs(true_spectrum) ** 2)
    energy_pred = np.sum(np.abs(pred_spectrum) ** 2)

    if energy_true == 0 and energy_pred == 0:
        energy_ratio = 1.0
    elif energy_true == 0:
        energy_ratio = (
            float("inf") if energy_pred > 1e-9 else 1.0
        )  # 如果预测能量也接近0，视为1
    else:
        energy_ratio = energy_pred / energy_true

    return {
        "Freq_MAE": freq_mae,
        "Freq_MSE": freq_mse,
        "Freq_RMSE": freq_rmse,
        "Freq_Correlation": corr,
        "Energy_Ratio": energy_ratio,
        "MSE": np.mean((pred_spectrum - true_spectrum) ** 2),
        "MAE": np.mean(np.abs(pred_spectrum - true_spectrum)),
    }


def plot_frequency_band_energy_custom_bar_chart(
    mean_energy_true,
    mean_energy_model1,
    mean_energy_model2,  # Now using mean energies
    band_names,
    model1_name,
    model2_name_suffix,
    feature_id_for_save,
    save_dir,
):
    """
    为平均频带能量绘制新的条形图，显示SRRF增加的能量。
    左柱: Ground Truth
    右柱: Model 1 (base), 在其上堆叠 (Model 2 - Model 1) 的能量增量
    """
    num_bands = len(band_names)
    x = np.arange(num_bands)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    # 1. 左边的柱子: Ground Truth (Average)
    rects1 = ax.bar(
        x - width / 2,
        mean_energy_true,
        width,
        label="Ground Truth (Average)",
        color="#FF8C00",
    )  # DarkOrange

    # 2. 右边的柱子: Model 1 (Average) + Delta for Model 2 (Average)
    rects2_base = ax.bar(
        x + width / 2,
        mean_energy_model1,
        width,
        label=f"{model1_name} (Average)",
        color="deepskyblue",
    )

    delta_energy_m2_vs_m1 = mean_energy_model2 - mean_energy_model1

    # 根据增量正负选择颜色
    delta_colors = [
        "mediumseagreen" if d >= 0 else "lightcoral" for d in delta_energy_m2_vs_m1
    ]

    rects2_delta = ax.bar(
        x + width / 2,
        delta_energy_m2_vs_m1,
        width,
        bottom=mean_energy_model1,
        label=f"Delta by {model2_name_suffix} (Average)",
        color=delta_colors,
    )

    ax.set_ylabel("Average Energy in Band (PSD sum)", fontsize=12)
    ax.set_xlabel("Frequency Band", fontsize=12)
    ax.set_title(
        f"Average Frequency Band Energy - Feature {feature_id_for_save}\n(Ground Truth vs. {model1_name} & {model1_name}{model2_name_suffix} Delta)",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 标注函数，用于在柱子顶部添加数值
    def autolabel(rects, values, is_delta=False, base_values=None):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            val_to_display = values[i]

            if is_delta:
                # 增量标注
                y_pos = (
                    base_values[i] + height / 2
                    if height >= 0
                    else base_values[i] + height + height / 2
                )  # 尝试放在delta条中间
                # 进一步调整，确保不与基底重叠且可见
                if abs(height) < 0.01 * ax.get_ylim()[1]:  # 如果delta太小
                    y_pos = base_values[i] + (
                        0.02 * ax.get_ylim()[1]
                        if height >= 0
                        else -0.02 * ax.get_ylim()[1]
                    )

                # 确保标注在图内
                if y_pos > ax.get_ylim()[1] * 0.95:
                    y_pos = ax.get_ylim()[1] * 0.95
                if y_pos < ax.get_ylim()[0] + 0.05 * ax.get_ylim()[1]:
                    y_pos = ax.get_ylim()[0] + 0.05 * ax.get_ylim()[1]
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    y_pos,
                    f"{val_to_display:+.4f}",  # 非科学计数法，4位小数
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                    bbox=(
                        dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5)
                        if abs(val_to_display) > 1e-3
                        else None
                    ),
                )  # 小值加背景
            else:
                # Ground Truth 和 Model 1 基础部分的标注
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height + (0.01 * ax.get_ylim()[1]),
                    f"{val_to_display:.3f}",  # 非科学计数法，3位小数
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    autolabel(rects1, mean_energy_true)  # 标注 Ground Truth 柱子
    autolabel(rects2_base, mean_energy_model1)  # 标注 Model 1 基础柱子
    autolabel(
        rects2_delta,
        delta_energy_m2_vs_m1,
        is_delta=True,
        base_values=mean_energy_model1,
    )  # 标注增量

    fig.tight_layout()
    plot_filename = f"{save_dir}/frequency_band_energy_custom_bars_AVERAGE_feature{feature_id_for_save}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    print(f"  平均频带能量自定义条形图已保存: {plot_filename}")


def plot_frequency_band_energy(
    true_values_all_samples,  # 特定特征下的所有样本 (num_samples, sequence_length)
    model1_pred_all_samples,  # 特定特征下的所有样本
    model2_pred_all_samples,  # 特定特征下的所有样本
    best_m2_sample_id_global,  # 模型2全局最优表现的样本ID
    model1_name="iTransformer",
    model2_name="+SRRF",
    sampling_rate=1.0,
    freq_unit_label_short="cycles/hour",
    feature_id_for_save="unknown",  # 用于保存文件名的特征ID
    save_dir="freq_results",
):
    """分析特定特征的频带能量分布，记录平均值和模型2全局最佳样本的情况"""
    num_bands = 5
    nyquist_freq = sampling_rate / 2
    bands = np.linspace(0, nyquist_freq, num_bands + 1)
    band_names = [
        f"{bands[i]:.2f}-{bands[i+1]:.2f} ({freq_unit_label_short})"
        for i in range(num_bands)
    ]

    num_samples = true_values_all_samples.shape[0]

    sample_energies_true = np.zeros((num_samples, num_bands))
    sample_energies_model1 = np.zeros((num_samples, num_bands))
    sample_energies_model2 = np.zeros((num_samples, num_bands))

    for i in range(num_samples):
        f_true, Pxx_true = compute_power_spectral_density(
            true_values_all_samples[i, :], fs=sampling_rate
        )
        f_model1, Pxx_model1 = compute_power_spectral_density(
            model1_pred_all_samples[i, :], fs=sampling_rate
        )
        f_model2, Pxx_model2 = compute_power_spectral_density(
            model2_pred_all_samples[i, :], fs=sampling_rate
        )

        for j in range(num_bands):
            band_start, band_end = bands[j], bands[j + 1]

            if len(f_true) > 0:
                idx_true = np.logical_and(f_true >= band_start, f_true < band_end)
                sample_energies_true[i, j] = (
                    np.sum(Pxx_true[idx_true]) if np.any(idx_true) else 0
                )
            if len(f_model1) > 0:
                idx_model1 = np.logical_and(f_model1 >= band_start, f_model1 < band_end)
                sample_energies_model1[i, j] = (
                    np.sum(Pxx_model1[idx_model1]) if np.any(idx_model1) else 0
                )
            if len(f_model2) > 0:
                idx_model2 = np.logical_and(f_model2 >= band_start, f_model2 < band_end)
                sample_energies_model2[i, j] = (
                    np.sum(Pxx_model2[idx_model2]) if np.any(idx_model2) else 0
                )

    # 1. 计算并保存该特征下所有样本的平均频带能量
    mean_energy_true = np.mean(sample_energies_true, axis=0)
    mean_energy_model1 = np.mean(sample_energies_model1, axis=0)
    mean_energy_model2 = np.mean(sample_energies_model2, axis=0)

    avg_energy_data = {
        "Frequency Band": band_names,
        "Avg Ground Truth Energy": mean_energy_true,
        f"Avg {model1_name} Energy": mean_energy_model1,
        f"Avg {model2_name} Energy": mean_energy_model2,
        f"Avg {model1_name}/GT Ratio": mean_energy_model1 / (mean_energy_true + 1e-9),
        f"Avg {model2_name}/GT Ratio": mean_energy_model2 / (mean_energy_true + 1e-9),
    }
    avg_energy_df = pd.DataFrame(avg_energy_data)
    print(f"\n--- 特征 {feature_id_for_save}: 平均频带能量 ---")
    print(avg_energy_df)
    avg_energy_df.to_csv(
        f"{save_dir}/frequency_band_energy_feature{feature_id_for_save}_AVERAGE.csv",
        index=False,
    )

    # 2. 记录模型2全局最佳样本在该特征下的频带能量
    if 0 <= best_m2_sample_id_global < num_samples:
        best_sample_true_energy = sample_energies_true[best_m2_sample_id_global, :]
        best_sample_model1_energy = sample_energies_model1[best_m2_sample_id_global, :]
        best_sample_model2_energy = sample_energies_model2[best_m2_sample_id_global, :]

        best_sample_energy_data = {
            "Frequency Band": band_names,
            f"GT Energy (Sample {best_m2_sample_id_global})": best_sample_true_energy,
            f"{model1_name} Energy (Sample {best_m2_sample_id_global})": best_sample_model1_energy,
            f"{model2_name} Energy (Sample {best_m2_sample_id_global})": best_sample_model2_energy,
            f"{model1_name}/GT Ratio (Sample {best_m2_sample_id_global})": best_sample_model1_energy
            / (best_sample_true_energy + 1e-9),
            f"{model2_name}/GT Ratio (Sample {best_m2_sample_id_global})": best_sample_model2_energy
            / (best_sample_true_energy + 1e-9),
        }
        best_sample_energy_df = pd.DataFrame(best_sample_energy_data)
        print(
            f"\n--- 特征 {feature_id_for_save}: 模型2全局最佳样本 (ID {best_m2_sample_id_global}) 的频带能量 ---"
        )
        print(best_sample_energy_df)
        best_sample_energy_df.to_csv(
            f"{save_dir}/frequency_band_energy_feature{feature_id_for_save}_MODEL2_BEST_SAMPLE_ID{best_m2_sample_id_global}.csv",
            index=False,
        )
    else:
        print(
            f"警告: 模型2的全局最佳样本ID {best_m2_sample_id_global} 无效，无法记录其在该特征下的频带能量。"
        )

    plot_frequency_band_energy_custom_bar_chart(
        mean_energy_true,
        mean_energy_model1,
        mean_energy_model2,
        band_names,
        model1_name,
        model2_name,  # Pass model1_name and model2_name (suffix)
        feature_id_for_save,
        save_dir,
    )
    return avg_energy_df


def plot_frequency_domain_metrics_heatmap(
    true_values,
    model1_pred,
    model2_pred,
    model1_name="iTransformer",
    model2_name="+SRRF",
    save_dir="freq_results",
):
    """绘制频域指标热力图 (所有特征的平均频谱指标)"""
    num_features = true_values.shape[2]
    metrics_to_plot = ["Freq_MAE", "Freq_RMSE", "Freq_Correlation", "Energy_Ratio"]
    results = np.zeros((len(metrics_to_plot), num_features, 2))

    for f_idx in range(num_features):
        all_true_spec_f, all_model1_spec_f, all_model2_spec_f = [], [], []
        for s_idx in range(true_values.shape[0]):
            _, mag_true = compute_fft(true_values[s_idx, :, f_idx])
            _, mag_model1 = compute_fft(model1_pred[s_idx, :, f_idx])
            _, mag_model2 = compute_fft(model2_pred[s_idx, :, f_idx])
            if len(mag_true) > 0:
                all_true_spec_f.append(mag_true)  # 只有在非空时添加
            if len(mag_model1) > 0:
                all_model1_spec_f.append(mag_model1)
            if len(mag_model2) > 0:
                all_model2_spec_f.append(mag_model2)

        # 确保列表非空才计算平均值
        avg_true_spec = (
            np.mean(all_true_spec_f, axis=0)
            if len(all_true_spec_f) > 0
            else np.array([])
        )
        avg_model1_spec = (
            np.mean(all_model1_spec_f, axis=0)
            if len(all_model1_spec_f) > 0
            else np.array([])
        )
        avg_model2_spec = (
            np.mean(all_model2_spec_f, axis=0)
            if len(all_model2_spec_f) > 0
            else np.array([])
        )

        model1_metrics = calculate_frequency_domain_metrics(
            avg_true_spec, avg_model1_spec
        )
        model2_metrics = calculate_frequency_domain_metrics(
            avg_true_spec, avg_model2_spec
        )

        for m_idx, metric_name in enumerate(metrics_to_plot):
            results[m_idx, f_idx, 0] = model1_metrics.get(metric_name, float("nan"))
            results[m_idx, f_idx, 1] = model2_metrics.get(metric_name, float("nan"))

    fig_height = max(12, 4 * len(metrics_to_plot))
    fig, axes = plt.subplots(
        len(metrics_to_plot),
        1,
        figsize=(max(12, num_features * 0.8), fig_height),
        squeeze=False,
    )  # 调整宽度以适应特征数量
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        data_for_heatmap = results[i, :, :]
        if metric == "Freq_Correlation":
            vmin, vmax, cmap = -1, 1, "RdBu_r"
        elif metric == "Energy_Ratio":
            abs_max_dev_from_1 = (
                np.nanmax(np.abs(data_for_heatmap[~np.isinf(data_for_heatmap)] - 1.0))
                if np.any(~np.isnan(data_for_heatmap) & ~np.isinf(data_for_heatmap))
                else 0.1
            )
            vmin, vmax, cmap = (
                1 - abs_max_dev_from_1,
                1 + abs_max_dev_from_1,
                "coolwarm",
            )
            if (
                np.isnan(vmin)
                or np.isinf(vmin)
                or np.isnan(vmax)
                or np.isinf(vmax)
                or vmin == vmax
            ):
                vmin, vmax = 0, 2
        else:  # MAE, RMSE
            all_valid_data = data_for_heatmap[~np.isnan(data_for_heatmap)]
            vmin = np.min(all_valid_data) if len(all_valid_data) > 0 else 0
            vmax = np.max(all_valid_data) if len(all_valid_data) > 0 else 1
            cmap = "coolwarm_r"
            if vmin == vmax:
                vmin = 0 if vmax > 0 else -1
                vmax = vmax + 1 if vmax > 0 else 0  # 避免vmin=vmax

        sns.heatmap(
            data_for_heatmap.T,
            ax=axes[i],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            yticklabels=[model1_name, model2_name],
            xticklabels=[
                f"F{j}" for j in range(num_features)
            ],  # 使用更短的标签 F0, F1...
            annot=True,
            fmt=".3f",
            cbar_kws={"label": metric},
        )
        axes[i].set_title(f"{metric} Comparison: Models vs Features")
        axes[i].set_xlabel("Features")
        axes[i].set_ylabel("Models")

    plt.tight_layout(pad=2.0)  # 增加一点padding
    heatmap_filename = f"{save_dir}/frequency_domain_metrics_heatmap.png"
    plt.savefig(heatmap_filename, dpi=300)
    plt.close()
    print(f"  频域指标热力图已保存: {heatmap_filename}")

    metrics_data_list = []
    for f_idx in range(num_features):
        row_m1 = {"Model": model1_name, "Feature": f"Feature {f_idx}"}
        row_m2 = {"Model": model2_name, "Feature": f"Feature {f_idx}"}
        for m_idx, metric_name in enumerate(metrics_to_plot):
            row_m1[metric_name] = results[m_idx, f_idx, 0]
            row_m2[metric_name] = results[m_idx, f_idx, 1]
        metrics_data_list.append(row_m1)
        metrics_data_list.append(row_m2)

    metrics_df = pd.DataFrame(metrics_data_list)
    metrics_df_filename = (
        f"{save_dir}/frequency_domain_metrics_all_features_AVERAGE_SPECTRUM.csv"
    )
    metrics_df.to_csv(metrics_df_filename, index=False)
    print(f"  所有特征的平均频谱指标已保存到CSV: {metrics_df_filename}")
    return results


def main():

    save_dir = ensure_dir("freq_results")
    true_values, model1_pred, model2_pred = load_sample_data(
        path="results/long_term_forecast_Exchange_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh1_el1_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
    )
    model1_name = "Base Model"
    model2_name = "+SRRF"

    num_samples = true_values.shape[0]
    num_features = true_values.shape[2]

    # --- 1. 找到模型2全局最优的 (样本ID, 特征ID) ---
    best_m2_overall_rmse = float("inf")
    best_m2_overall_sample_id = -1
    best_m2_overall_feature_id = -1
    bset_mse = float("inf")

    print("开始搜索模型2的全局最优表现点 (基于频谱RMSE)...")
    for s_id in range(num_samples):
        for f_id in range(num_features):
            _, true_sample_fft_mag = compute_fft(true_values[s_id, :, f_id])
            _, model2_sample_fft_mag = compute_fft(model2_pred[s_id, :, f_id])

            if len(true_sample_fft_mag) == 0 or len(model2_sample_fft_mag) == 0:
                continue

            metrics_m2 = calculate_frequency_domain_metrics(
                true_sample_fft_mag, model2_sample_fft_mag
            )
            current_rmse_m2 = metrics_m2.get("Freq_RMSE", float("inf"))
            # current_rmse_m2 = np.mean((true_values[s_id, :, f_id] - model2_pred[s_id, :, f_id]) ** 2)
            # current_rmse_m2 = np.mean(
            #     np.abs(true_values[s_id, :, f_id] - model2_pred[s_id, :, f_id])
            # )

            if not np.isnan(current_rmse_m2) and current_rmse_m2 < best_m2_overall_rmse:
                best_m2_overall_rmse = current_rmse_m2
                best_m2_overall_sample_id = s_id
                best_m2_overall_feature_id = f_id
        if (s_id + 1) % (
            num_samples // 10 if num_samples >= 10 else 1
        ) == 0:  # 打印进度
            print(f"  已扫描 {s_id + 1}/{num_samples} 个样本...")

    if best_m2_overall_sample_id == -1:
        print("错误：未能找到模型2的有效最佳表现点。请检查数据和计算过程。脚本将退出。")
        return

    print("\n--- 模型2 全局最优表现点 (基于频谱RMSE) ---")
    print(f"  最佳样本ID: {best_m2_overall_sample_id}")
    print(f"  最佳特征ID: {best_m2_overall_feature_id}")
    print(f"  对应的模型2 Freq_RMSE: {best_m2_overall_rmse:.4f}")
    print("-------------------------------------------------")

    # --- 2. sampling_rate 设置 ---
    sampling_rate = 1.0  # 假设ETTh1每小时采样，频率单位为 周期/小时
    freq_unit_label = "Frequency (cycles/hour)"
    freq_unit_label_short = "cyc/hr"
    print(
        f"\n使用 sampling_rate = {sampling_rate} ({freq_unit_label_short}) 进行后续分析。"
    )

    # --- 3. 针对模型2最优表现点进行详细分析和绘图 ---
    print(
        f"\n开始针对模型2最优表现点 (样本 {best_m2_overall_sample_id}, 特征 {best_m2_overall_feature_id}) 进行详细分析..."
    )

    # 提取该最优点的具体数据
    true_sample_best_m2 = true_values[
        best_m2_overall_sample_id, :, best_m2_overall_feature_id
    ]
    model1_pred_best_m2 = model1_pred[
        best_m2_overall_sample_id, :, best_m2_overall_feature_id
    ]
    model2_pred_best_m2 = model2_pred[
        best_m2_overall_sample_id, :, best_m2_overall_feature_id
    ]
    model_input = model1_pred[
        best_m2_overall_sample_id - 1, :, best_m2_overall_feature_id
    ]

    #   a. 绘制频谱图
    plot_frequency_spectrum(
        model_input,
        true_sample_best_m2,
        model1_pred_best_m2,
        model2_pred_best_m2,
        sample_id_for_title=best_m2_overall_sample_id,
        feature_id_for_title=best_m2_overall_feature_id,
        model1_name=model1_name,
        model2_name=model2_name,
        sampling_rate=sampling_rate,
        freq_unit_label=freq_unit_label,
        save_dir=save_dir,
        filename_suffix="_MODEL2_BEST_CASE",
    )

    #   b. 绘制PSD图
    plot_power_spectral_density(
        true_sample_best_m2,
        model1_pred_best_m2,
        model2_pred_best_m2,
        sample_id_for_title=best_m2_overall_sample_id,
        feature_id_for_title=best_m2_overall_feature_id,
        model1_name=model1_name,
        model2_name=model2_name,
        sampling_rate=sampling_rate,
        freq_unit_label=freq_unit_label,
        save_dir=save_dir,
        filename_suffix="_MODEL2_BEST_CASE",
    )

    #   c. 计算并打印该最优点的详细频域指标对比
    print(
        f"\n--- 模型2最优表现点 (样本 {best_m2_overall_sample_id}, 特征 {best_m2_overall_feature_id}) 的频域指标对比 ---"
    )
    _, true_fft_mag_best_m2 = compute_fft(true_sample_best_m2)
    _, m1_fft_mag_best_m2 = compute_fft(model1_pred_best_m2)
    _, m2_fft_mag_best_m2 = compute_fft(model2_pred_best_m2)

    metrics_model1_at_best_m2_case = calculate_frequency_domain_metrics(
        true_fft_mag_best_m2, m1_fft_mag_best_m2
    )
    metrics_model2_at_best_m2_case = calculate_frequency_domain_metrics(
        true_fft_mag_best_m2, m2_fft_mag_best_m2
    )

    metrics_comparison_data = {
        "Metric": list(metrics_model1_at_best_m2_case.keys()),
        model1_name: list(metrics_model1_at_best_m2_case.values()),
        model2_name: list(metrics_model2_at_best_m2_case.values()),
    }
    metrics_comparison_df = pd.DataFrame(metrics_comparison_data)
    print(metrics_comparison_df)
    metrics_comparison_df.to_csv(
        f"{save_dir}/metrics_comparison_model2_best_case_s{best_m2_overall_sample_id}_f{best_m2_overall_feature_id}.csv",
        index=False,
    )
    print(
        f"  指标对比已保存到: {save_dir}/metrics_comparison_model2_best_case_s{best_m2_overall_sample_id}_f{best_m2_overall_feature_id}.csv"
    )
    print(
        "---------------------------------------------------------------------------------"
    )

    # --- 4. 频带能量分析 (针对模型2最优表现的特征ID) ---
    print(
        f"\n开始针对模型2最优表现点对应的特征 (特征 {best_m2_overall_feature_id}) 进行频带能量分析..."
    )
    plot_frequency_band_energy(
        true_values[:, :, best_m2_overall_feature_id],  # 该特征下的所有样本
        model1_pred[:, :, best_m2_overall_feature_id],
        model2_pred[:, :, best_m2_overall_feature_id],
        best_m2_sample_id_global=best_m2_overall_sample_id,  # 传入模型2全局最优样本ID
        model1_name=model1_name,
        model2_name=model2_name,
        sampling_rate=sampling_rate,
        freq_unit_label_short=freq_unit_label_short,
        feature_id_for_save=str(best_m2_overall_feature_id),  # 使用特征ID命名文件
        save_dir=save_dir,
    )

    # --- 5. 频域指标热力图 (所有特征的平均频谱指标，作为概览) ---
    print("\n开始生成所有特征的平均频域指标热力图 (概览)...")
    plot_frequency_domain_metrics_heatmap(
        true_values,
        model1_pred,
        model2_pred,
        model1_name=model1_name,
        model2_name=model2_name,
        save_dir=save_dir,
    )

    print(f"\n所有频域分析结果已保存到 {save_dir}/")
    print("分析完成!")


if __name__ == "__main__":
    main()
