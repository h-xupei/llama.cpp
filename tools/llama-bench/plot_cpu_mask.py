# plot_cpu_mask_compare.py
# 读取 cpu_mask_compare.csv，绘制：
# 1) 频率 vs steady token/s
# 2) 频率 vs FTL
# 3) 频率 vs 最大温度
# 4) 频率 vs 单位Token能耗 (mJ/token)
#
# 用法：
# python plot_cpu_mask_compare.py
#
# 默认要求当前目录下存在：
#   cpu_mask_compare.csv

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("plot_data_cpu_mask_refine.csv")


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {CSV_PATH.resolve()}")

    # 优先按 utf-8 读，失败再尝试 gbk / utf-8-sig
    encodings = ["utf-8", "utf-8-sig", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"读取 CSV 失败，最后错误: {last_err}")

    required_cols = [
        "cpu_mask",
        "real_freq_khz",
        "freq_mhz",
        "n_threads",
        "avg_steady_ts",
        "avg_ftl_s",
        "window_max_temp_c",
        "avg_mJ_per_token",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    # 只画 n_threads=3
    df = df[df["n_threads"] == 3].copy()
    if df.empty:
        raise ValueError("筛选 n_threads == 3 后无数据")

    # 统一排序
    df = df.sort_values(["cpu_mask", "freq_mhz"]).reset_index(drop=True)
    return df


def draw_metric(df: pd.DataFrame, y_col: str, y_label: str, title: str, out_name: str) -> None:
    plt.figure(figsize=(8, 5))

    for mask in sorted(df["cpu_mask"].unique()):
        sub = df[df["cpu_mask"] == mask].sort_values("freq_mhz")
        plt.plot(
            sub["freq_mhz"],
            sub[y_col],
            marker="o",
            linewidth=2,
            label=mask,
        )

    plt.xlabel("Frequency (MHz)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="CPU Mask")
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"已生成: {out_name}")


def main() -> None:
    df = load_data()

    draw_metric(
        df=df,
        y_col="avg_steady_ts",
        y_label="Steady token/s",
        title="CPU Mask Comparison: Frequency vs Steady token/s",
        out_name="compare_freq_vs_steady_tokens.png",
    )

    draw_metric(
        df=df,
        y_col="avg_ftl_s",
        y_label="FTL (s)",
        title="CPU Mask Comparison: Frequency vs First Token Latency",
        out_name="compare_freq_vs_ftl.png",
    )

    draw_metric(
        df=df,
        y_col="window_max_temp_c",
        y_label="Max Temperature (°C)",
        title="CPU Mask Comparison: Frequency vs Max Temperature",
        out_name="compare_freq_vs_max_temp.png",
    )

    draw_metric(
        df=df,
        y_col="avg_mJ_per_token",
        y_label="Energy per Token (mJ/token)",
        title="CPU Mask Comparison: Frequency vs Energy per Token",
        out_name="compare_freq_vs_mj_per_token.png",
    )


if __name__ == "__main__":
    main()