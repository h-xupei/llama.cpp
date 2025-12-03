#pragma once

#include "cpu_freq_optimizer_base.cpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// 二维“邻域搜索”优化器：
// - 4 个直接邻居（上下左右）使用真实 cost（必须先实际跑过）
// - 4 个对角邻居的 cost 采用插值：
//      Cost(fi+x, nj+y) ≈ ( Cost(fi+x, nj) + Cost(fi, nj+y) ) / 2,  x,y ∈ {-1, 1}
// 策略：
//  1）优先探索未评估的 4 个直接邻居（cost == +inf）；
//  2）当当前点 + 所有直接邻居都有真实 cost 后：
//      - 直接邻居用真实 cost
//      - 对角邻居用插值 cost（若两条边都已测过）
//      - 在 {当前 + 所有直接邻居 + 有插值的对角邻居} 中选 cost 最小配置
//  3）如果 best 就是当前点，相当于平台驻留（不强制跳）
class NeighborSearchOptimizer : public CpuFreqOptimizerBase {
  public:
    NeighborSearchOptimizer(double                   alpha,
                            const std::vector<int> & freqLevels,
                            const std::vector<int> & threadLevels,
                            size_t                   cacheLength) :
        CpuFreqOptimizerBase(alpha, freqLevels, threadLevels, cacheLength) {
        // 起点：最高频 + 最大线程（基类已设置）
    }

  protected:
    void chooseNextConfig() override {
        CpuFreqConfig cur = currentConfig();
        int           f   = cur.freqIdx;
        int           t   = cur.threadIdx;

        struct Neighbor {
            CpuFreqConfig cfg;
            double        cost;
            bool          approx;  // 是否插值 cost
        };

        std::vector<Neighbor> candidates;

        // 先把当前点放进去（真实 cost）
        double curCost = costOf(f, t);
        candidates.push_back({ cur, curCost, false });

        // ================== 1. 4 个直接邻居 ==================
        const int df[4] = { -1, 1, 0, 0 };  // f 偏移
        const int dt[4] = { 0, 0, -1, 1 };  // t 偏移

        std::vector<CpuFreqConfig> directNeighbors;
        directNeighbors.reserve(4);

        for (int k = 0; k < 4; ++k) {
            int nf = f + df[k];
            int nt = t + dt[k];
            if (validIdx(nf, nt)) {
                directNeighbors.push_back({ nf, nt });
            }
        }

        std::vector<CpuFreqConfig> unexploredDirect;

        for (const auto & nb : directNeighbors) {
            double c = costOf(nb.freqIdx, nb.threadIdx);
            if (std::isinf(c)) {
                // 还没评估过，优先探索
                unexploredDirect.push_back(nb);
            } else {
                // 已评估，加入候选池
                candidates.push_back({ nb, c, false });
            }
        }

        // 若还有没测过的直接邻居：先随机挑一个去探索
        if (!unexploredDirect.empty()) {
            std::uniform_int_distribution<size_t> dist(0, unexploredDirect.size() - 1);
            CpuFreqConfig                         pick = unexploredDirect[dist(rng_)];

            std::cout << "[NEIGH] exploring unseen direct neighbor: freqIdx=" << pick.freqIdx << " ("
                      << freqLevels_[pick.freqIdx] << " kHz), threadIdx=" << pick.threadIdx
                      << " (n=" << threadLevels_[pick.threadIdx] << ")\n";

            setCurrentConfig(pick.freqIdx, pick.threadIdx);
            return;
        }

        // 到这里说明：当前点 + 4 个直接邻居，都已经有真实 cost，可以尝试对角插值

        // ================== 2. 4 个对角邻居插值 ==================
        const int dx[2] = { -1, 1 };
        const int dy[2] = { -1, 1 };

        for (int ix = 0; ix < 2; ++ix) {
            for (int iy = 0; iy < 2; ++iy) {
                int nf = f + dx[ix];
                int nt = t + dy[iy];
                if (!validIdx(nf, nt)) {
                    continue;
                }

                // 对角插值需要两个“边”的真实 cost：
                //   (f + dx, t)   &   (f, t + dy)
                int f_edge_f = f + dx[ix];
                int f_edge_t = t;
                int t_edge_f = f;
                int t_edge_t = t + dy[iy];

                if (!validIdx(f_edge_f, f_edge_t) || !validIdx(t_edge_f, t_edge_t)) {
                    continue;
                }

                double cost_h = costOf(f_edge_f, f_edge_t);
                double cost_v = costOf(t_edge_f, t_edge_t);

                if (std::isinf(cost_h) || std::isinf(cost_v)) {
                    // 任一边尚未真实评估，则先不插值
                    continue;
                }

                double        approxCost = 0.5 * (cost_h + cost_v);
                CpuFreqConfig diagCfg{ nf, nt };
                candidates.push_back({ diagCfg, approxCost, true });
            }
        }

        // ================== 3. 在候选集中选 cost 最小 ==================
        Neighbor best = candidates.front();
        for (const auto & cand : candidates) {
            if (cand.cost < best.cost) {
                best = cand;
            }
        }

        if (best.cfg.freqIdx == f && best.cfg.threadIdx == t) {
            // 最优点就是当前点：平台驻留，不强制跳
            std::cout << "[NEIGH] stay on current cfg (local minimum): freqIdx=" << f << " (" << freqLevels_[f]
                      << " kHz), threadIdx=" << t << " (n=" << threadLevels_[t] << ")"
                      << ", cost=" << best.cost << "\n";
        } else {
            std::cout << "[NEIGH] move to " << (best.approx ? "approx" : "real")
                      << " neighbor cfg: freqIdx=" << best.cfg.freqIdx << " (" << freqLevels_[best.cfg.freqIdx]
                      << " kHz), threadIdx=" << best.cfg.threadIdx << " (n=" << threadLevels_[best.cfg.threadIdx]
                      << "), cost=" << best.cost << "\n";
        }

        setCurrentConfig(best.cfg.freqIdx, best.cfg.threadIdx);
    }
};
