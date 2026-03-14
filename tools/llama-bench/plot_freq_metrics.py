# plot_freq_metrics.py
# 读取 freq_metrics_plot_data.csv，绘制：
# 1) 频率 vs steady token/s
# 2) 频率 vs FTL
# 3) 频率 vs 最大温度
# 4) 频率 vs 单位Token能耗（mJ/token）
#
# 用法：
# python plot_freq_metrics.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("plot_data_freq_metrics.csv")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {CSV_PATH.resolve()}")

    # 兼容 UTF-8 / UTF-8 BOM / GBK
    encodings = ["utf-8", "utf-8-sig", "gbk"]
    df = None
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc)
            print(f"成功读取 CSV，编码: {enc}")
            break
        except Exception as e:
            last_err = e

    if df is None:
        raise RuntimeError(f"读取 CSV 失败，最后错误: {last_err}")

    required_cols = [
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

    # 排序，保证画线时频率递增
    df = df.sort_values(["n_threads", "freq_mhz"]).reset_index(drop=True)

    # 线程数列表
    thread_list = sorted(df["n_threads"].unique())

    def draw_one(y_col: str, y_label: str, title: str, out_name: str):
        plt.figure(figsize=(8, 5))

        for n in thread_list:
            sub = df[df["n_threads"] == n].sort_values("freq_mhz")
            plt.plot(
                sub["freq_mhz"],
                sub[y_col],
                marker="o",
                linewidth=2,
                label=f"{int(n)} threads",
            )

        plt.xlabel("Frequency (MHz)")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_name, dpi=300)
        plt.close()
        print(f"已生成: {out_name}")

    draw_one(
        y_col="avg_steady_ts",
        y_label="Steady token/s",
        title="Frequency vs Steady token/s",
        out_name="freq_vs_steady_tokens.png",
    )

    draw_one(
        y_col="avg_ftl_s",
        y_label="FTL (s)",
        title="Frequency vs First Token Latency",
        out_name="freq_vs_ftl.png",
    )

    draw_one(
        y_col="window_max_temp_c",
        y_label="Max Temperature (°C)",
        title="Frequency vs Max Temperature",
        out_name="freq_vs_max_temp.png",
    )

    draw_one(
        y_col="avg_mJ_per_token",
        y_label="Energy per Token (mJ/token)",
        title="Frequency vs Energy per Token",
        out_name="freq_vs_mj_per_token.png",
    )


if __name__ == "__main__":
    main()
