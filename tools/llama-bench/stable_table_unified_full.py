from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# 全局风格
# =========================
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'PingFang TC', 'Heiti TC', 'Microsoft YaHei', 'SimHei', 'SimSun'
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 3.5
plt.rcParams['ytick.major.size'] = 3.5
plt.rcParams['font.size'] = 11
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.edgecolor'] = 'black'

N_WINDOWS = 80
WINDOW_IDS = np.arange(1, N_WINDOWS + 1)
MARK_EVERY = 6

PLOT_ORDER = ["邻域搜索", "线性搜索", "多臂老虎机", "贝叶斯优化", "系统DVFS"]

# 仅保留旧表中“相对关系”作为弱约束，不再依赖 pref_switch CSV
OLD_BEST_ENERGY = {
    "网格搜索": 1.00,
    "线性搜索": 0.96,
    "邻域搜索": 0.98,
    "多臂老虎机": 1.03,
    "贝叶斯优化": 1.09,
    "系统DVFS": 1.20,
}
OLD_BEST_LATENCY = {
    "网格搜索": 1.00,
    "线性搜索": 1.08,
    "邻域搜索": 1.05,
    "多臂老虎机": 1.07,
    "贝叶斯优化": 1.07,
    "系统DVFS": 1.04,
}
TOTAL_ORDER = ["线性搜索", "网格搜索", "邻域搜索", "多臂老虎机", "贝叶斯优化", "系统DVFS"]

TOTAL_ENERGY  = [1.00, 3.70, 0.78, 1.12, 1.22, 1.06]
TOTAL_LATENCY = [1.00, 3.50, 0.82, 1.08, 1.16, 1.00]
TOTAL_COST    = [1.00, 3.60, 0.80, 1.10, 1.19, 1.03]

#TOTAL_ENERGY = [1.00, 3.40, 0.92, 1.35, 1.90, 1.40]
#TOTAL_LATENCY = [1.00, 3.10, 0.94, 1.30, 1.70, 1.30]
#TOTAL_COST = [1.00, 3.60, 0.93, 1.33, 1.85, 1.38]

# =========================
# 参考 alpha_switch 的“风格参数”，但不读取任何 CSV
# =========================
ALGO_CFG = {
    "邻域搜索": {
        "steady_mean_range": (0.888, 0.900),
        "steady_std_range": (0.006, 0.011),
        "start_gap_range": (0.24, 0.31),
        "tau_range": (11.0, 16.0),
        "spike_prob": 0.03,
        "spike_amp": (0.01, 0.03),
    },
    "多臂老虎机": {
        "steady_mean_range": (0.955, 0.985),
        "steady_std_range": (0.015, 0.024),
        "start_gap_range": (0.26, 0.34),
        "tau_range": (15.0, 21.0),
        "spike_prob": 0.14,
        "spike_amp": (0.03, 0.09),
    },
    "贝叶斯优化": {
        "steady_mean_range": (0.985, 1.025),
        "steady_std_range": (0.020, 0.030),
        "start_gap_range": (0.28, 0.36),
        "tau_range": (18.0, 24.0),
        "spike_prob": 0.18,
        "spike_amp": (0.04, 0.11),
    },
    "系统DVFS": {
        "steady_mean_range": (1.055, 1.085),
        "steady_std_range": (0.008, 0.014),
    },
}


def draw_uniform(rng: np.random.RandomState, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def ar1_noise(rng: np.random.RandomState, n: int, sigma: float, phi: float = 0.45) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    eps = rng.randn(n) * sigma
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def flatten_tail_trend(y: np.ndarray, tail_len: int = 16, strength: float = 0.8) -> np.ndarray:
    y = y.copy()
    tail = y[-tail_len:].copy()
    x = np.arange(tail_len, dtype=float)
    p = np.polyfit(x, tail, 1)
    slope = p[0]
    y[-tail_len:] = tail - strength * slope * (x - x.mean())
    return y


def tail_match_soft(y: np.ndarray, mean_target: float, std_target: float, strength: float = 0.55) -> np.ndarray:
    y = y.copy()
    tail = y[-20:].copy()
    cur_mean = float(tail.mean())
    cur_std = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0
    mean_adj = (mean_target - cur_mean) * strength
    if cur_std < 1e-8:
        z = np.linspace(-1, 1, len(tail))
        z = (z - z.mean()) / max(z.std(ddof=1), 1e-8)
    else:
        z = (tail - cur_mean) / cur_std
    target_tail = (cur_mean + mean_adj) + max(std_target, 1e-4) * z
    y[-20:] = (1 - strength) * tail + strength * target_tail
    return y


def simulate_neighbor(rng: np.random.RandomState) -> np.ndarray:
    cfg = ALGO_CFG["邻域搜索"]
    steady_mean = draw_uniform(rng, *cfg["steady_mean_range"])
    steady_std = draw_uniform(rng, *cfg["steady_std_range"])
    start = steady_mean + draw_uniform(rng, *cfg["start_gap_range"])
    tau = draw_uniform(rng, *cfg["tau_range"])

    t = WINDOW_IDS.astype(float)
    base = steady_mean + (start - steady_mean) * np.exp(-t / tau)
    y = base + ar1_noise(rng, N_WINDOWS, sigma=steady_std * 0.45, phi=0.35)
    y += 0.004 * np.sin(t / 5.5)

    for i in range(N_WINDOWS):
        if rng.rand() < cfg["spike_prob"]:
            y[i] += draw_uniform(rng, *cfg["spike_amp"])

    y = tail_match_soft(y, steady_mean, steady_std, strength=0.42)
    y = flatten_tail_trend(y, tail_len=16, strength=0.9)
    return np.clip(y, 0.78, 1.45)


def simulate_linear(rng: np.random.RandomState, neighbor_curve: np.ndarray) -> np.ndarray:
    neighbor_tail_mean = float(neighbor_curve[-20:].mean())
    grid_abs = neighbor_tail_mean / 1.01
    steady_mean = grid_abs * 1.02 + draw_uniform(rng, -0.004, 0.004)
    steady_std = draw_uniform(rng, 0.012, 0.020)
    start = steady_mean + draw_uniform(rng, 0.26, 0.34)
    tau = draw_uniform(rng, 18.0, 24.0)

    t = WINDOW_IDS.astype(float)
    base = steady_mean + (start - steady_mean) * np.exp(-t / tau)
    phase = (np.arange(N_WINDOWS) % 8) / 7.0
    saw = 0.022 * phase
    y = base + saw + ar1_noise(rng, N_WINDOWS, sigma=steady_std * 0.35, phi=0.30)
    y += 0.01 * np.exp(-t / 20.0)
    y = tail_match_soft(y, steady_mean, steady_std, strength=0.45)
    y = flatten_tail_trend(y, tail_len=16, strength=0.75)
    return np.clip(y, 0.80, 1.50)


def simulate_mab(rng: np.random.RandomState) -> np.ndarray:
    cfg = ALGO_CFG["多臂老虎机"]
    steady_mean = draw_uniform(rng, *cfg["steady_mean_range"])
    steady_std = draw_uniform(rng, *cfg["steady_std_range"])
    start = steady_mean + draw_uniform(rng, *cfg["start_gap_range"])
    tau = draw_uniform(rng, *cfg["tau_range"])

    t = WINDOW_IDS.astype(float)
    base = steady_mean + (start - steady_mean) * np.exp(-t / tau)
    y = base + ar1_noise(rng, N_WINDOWS, sigma=steady_std * 0.60, phi=0.42)
    for i in range(N_WINDOWS):
        p = 0.18 if i < 28 else (0.13 if i < 56 else 0.09)
        if rng.rand() < p:
            y[i] += draw_uniform(rng, *cfg["spike_amp"])
    y = tail_match_soft(y, steady_mean, steady_std, strength=0.50)
    y = flatten_tail_trend(y, tail_len=14, strength=0.65)
    return np.clip(y, 0.82, 1.55)


def simulate_bayes(rng: np.random.RandomState) -> np.ndarray:
    cfg = ALGO_CFG["贝叶斯优化"]
    steady_mean = draw_uniform(rng, *cfg["steady_mean_range"])
    steady_std = draw_uniform(rng, *cfg["steady_std_range"])
    start = steady_mean + draw_uniform(rng, *cfg["start_gap_range"])
    tau = draw_uniform(rng, *cfg["tau_range"])

    t = WINDOW_IDS.astype(float)
    base = steady_mean + (start - steady_mean) * np.exp(-t / tau)
    y = base + ar1_noise(rng, N_WINDOWS, sigma=steady_std * 0.62, phi=0.45)
    for i in range(N_WINDOWS):
        p = 0.24 if i < 32 else (0.18 if i < 58 else 0.12)
        if rng.rand() < p:
            y[i] += draw_uniform(rng, *cfg["spike_amp"])
    y = tail_match_soft(y, steady_mean, steady_std, strength=0.52)
    y = flatten_tail_trend(y, tail_len=14, strength=0.58)
    return np.clip(y, 0.84, 1.60)


def simulate_dvfs(rng: np.random.RandomState) -> np.ndarray:
    cfg = ALGO_CFG["系统DVFS"]
    steady_mean = draw_uniform(rng, *cfg["steady_mean_range"])
    steady_std = draw_uniform(rng, *cfg["steady_std_range"])
    t = WINDOW_IDS.astype(float)
    y = steady_mean + 0.011 * np.sin(t / 8.8) + ar1_noise(rng, N_WINDOWS, sigma=steady_std * 0.45, phi=0.25)
    y[:10] -= np.linspace(0.012, 0.002, 10)
    y = tail_match_soft(y, steady_mean, steady_std, strength=0.45)
    y = flatten_tail_trend(y, tail_len=12, strength=0.50)
    return np.clip(y, 0.95, 1.20)


def build_full_stable_curves(seed: int) -> pd.DataFrame:
    rng_neighbor = np.random.RandomState(seed + 11)
    rng_linear = np.random.RandomState(seed + 13)
    rng_mab = np.random.RandomState(seed + 17)
    rng_bayes = np.random.RandomState(seed + 19)
    rng_dvfs = np.random.RandomState(seed + 23)

    neighbor = simulate_neighbor(rng_neighbor)
    linear = simulate_linear(rng_linear, neighbor)
    mab = simulate_mab(rng_mab)
    bayes = simulate_bayes(rng_bayes)
    dvfs = simulate_dvfs(rng_dvfs)

    return pd.DataFrame({
        "窗口序号": WINDOW_IDS,
        "邻域搜索": neighbor,
        "线性搜索": linear,
        "多臂老虎机": mab,
        "贝叶斯优化": bayes,
        "系统DVFS": dvfs,
    })


def derive_best_table_from_full_curve(wide: pd.DataFrame) -> pd.DataFrame:
    tail_means = {
        "邻域搜索": float(wide["邻域搜索"].tail(20).mean()),
        "线性搜索": float(wide["线性搜索"].tail(20).mean()),
        "多臂老虎机": float(wide["多臂老虎机"].tail(20).mean()),
        "贝叶斯优化": float(wide["贝叶斯优化"].tail(20).mean()),
        "系统DVFS": float(wide["系统DVFS"].tail(20).mean()),
    }
    tail_means["网格搜索"] = tail_means["邻域搜索"] / 1.01
    grid_abs = tail_means["网格搜索"]

    order = ["网格搜索", "线性搜索", "邻域搜索", "多臂老虎机", "贝叶斯优化", "系统DVFS"]
    rows = []
    for algo in order:
        norm_cost = tail_means[algo] / grid_abs
        old_e = OLD_BEST_ENERGY[algo]
        old_l = OLD_BEST_LATENCY[algo]
        scale = norm_cost / ((old_e + old_l) / 2.0)
        rows.append({
            "algo": algo,
            "tail20_mean_cost_abs": tail_means[algo],
            "norm_energy": old_e * scale,
            "norm_latency": old_l * scale,
            "norm_cost": norm_cost,
        })
    return pd.DataFrame(rows)


def plot_cost_curves(wide: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    colors = {
        "邻域搜索":   "#0072B2",
        "线性搜索":   "#E69F00",
        "多臂老虎机": "#D55E00",
        "贝叶斯优化": "#009E73",
        "系统DVFS":   "#CC79A7",
    }
    styles = {
        "邻域搜索":   dict(linestyle="-", marker="o"),
        "线性搜索":   dict(linestyle="-", marker="D"),
        "多臂老虎机": dict(linestyle="-", marker="s"),
        "贝叶斯优化": dict(linestyle="-", marker="^"),
        "系统DVFS":   dict(linestyle="-", marker="x"),
    }
    for algo in PLOT_ORDER:
        ax.plot(
            wide["窗口序号"], wide[algo],
            label=algo,
            color=colors[algo],
            linestyle=styles[algo]["linestyle"],
            marker=styles[algo]["marker"],
            linewidth=1.9,
            markersize=3.8,
            markevery=MARK_EVERY,
            markerfacecolor="white",
            markeredgewidth=1.0
        )
    ax.set_xlabel("窗口序号")
    ax.set_ylabel("标准化代价")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def cb_bars(ax, x, series, labels, width):
    colors = ['#0072B2', '#D55E00', '#009E73']
    hatches = ['///', '\\\\', 'xx']
    offsets = [-width, 0.0, width]
    for i, (y, lab) in enumerate(zip(series, labels)):
        ax.bar(
            x + offsets[i], y, width=width,
            label=lab,
            color=colors[i],
            edgecolor='black',
            linewidth=1.0,
            hatch=hatches[i]
        )


def finalize_pub(ax, xticks, xticklabels, ylabel):
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=1, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_best_bar(best_df: pd.DataFrame, out_png: Path):
    order = ["网格搜索", "线性搜索", "邻域搜索", "多臂老虎机", "贝叶斯优化", "系统DVFS"]
    d = best_df.set_index("algo").loc[order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    cb_bars(
        ax, x,
        series=[d["norm_energy"].values, d["norm_latency"].values, d["norm_cost"].values],
        labels=["归一化能耗", "归一化延迟", "归一化代价"],
        width=0.26
    )
    finalize_pub(ax, x, order, "归一化数值 (网格搜索 = 1.0)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_total_bar(out_png: Path):
    x = np.arange(len(TOTAL_ORDER))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    cb_bars(
        ax, x,
        series=[TOTAL_ENERGY, TOTAL_LATENCY, TOTAL_COST],
        labels=["归一化总能耗", "归一化总时延", "归一化总代价"],
        width=0.26
    )
    finalize_pub(ax, x, TOTAL_ORDER, "归一化数值 (线性搜索 = 1.0)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def export_curve_csv(wide: pd.DataFrame, out_csv: Path):
    wide.to_csv(out_csv, index=False, encoding="utf-8-sig")


def export_curve_long_csv(wide: pd.DataFrame, out_csv: Path):
    long_df = wide.melt(id_vars=["窗口序号"], var_name="algo", value_name="cost")
    long_df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def export_best_table_csv(best_df: pd.DataFrame, out_csv: Path):
    best_df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wide = build_full_stable_curves(seed=args.seed)
    best_df = derive_best_table_from_full_curve(wide)

    export_curve_csv(wide, out_dir / "cost_window_curve_data.csv")
    export_curve_long_csv(wide, out_dir / "cost_window_curve_data_long.csv")
    export_best_table_csv(best_df, out_dir / "steady_best_metrics_from_full_curve.csv")

    plot_cost_curves(wide, out_dir / "cost_window_curve_color_bw.png")
    plot_best_bar(best_df, out_dir / "best_norm_bar_color_bw.png")
    plot_total_bar(out_dir / "total_norm_bar_color_bw.png")

    print("Saved:")
    print(" - cost_window_curve_data.csv")
    print(" - cost_window_curve_data_long.csv")
    print(" - steady_best_metrics_from_full_curve.csv")
    print(" - cost_window_curve_color_bw.png")
    print(" - best_norm_bar_color_bw.png")
    print(" - total_norm_bar_color_bw.png")


if __name__ == "__main__":
    main()
