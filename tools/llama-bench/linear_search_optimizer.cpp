#pragma once

#include "cpu_freq_optimizer_base.cpp"

#include <iostream>
#include <limits>
#include <vector>

// 线性搜索（Linear Search）优化器：
// 参考 EnergyOptimizer_linearsweeps 的二维版本
//
// 流程：
//   lastSweepBackup 初始为 [starting_config]
//   1）下一 sweep 维度 = THREAD：
//       - 在 lastSweepBackup 中找 cost 最小配置 best_cfg
//       - 固定 best_cfg.freqIdx，在线程维度上扫所有 threadIdx，生成队列 sweepQueue_
//       - lastSweepBackup 更新为这一批线程 sweep 的所有配置
//   2）下一 sweep 维度 = FREQ：
//       - 在 lastSweepBackup 里找 cost 最好的 (freq,thread)
//       - 固定 threadIdx，在线频维度上扫所有 freqIdx
//   3）两维都扫完后：在整个 2D 空间 historyCost_ 里找全局最优，之后一直用它
class LinearSearchOptimizer : public CpuFreqOptimizerBase {
  public:
    LinearSearchOptimizer(double                   alpha,
                          const std::vector<int> & freqLevels,
                          const std::vector<int> & threadLevels,
                          size_t                   cacheLength) :
        CpuFreqOptimizerBase(alpha, freqLevels, threadLevels, cacheLength),
        lastSweep_(Dim::THREAD),  // 参照 Python 默认 last_sweep="BATCHSIZE" -> next 是 CPU
        numSweepsDone_(0),
        finished_(false) {
        // 起点从基类：最高频+最大线程
        lastSweepBackup_.push_back(currentConfig());
    }

  protected:
    void chooseNextConfig() override {
        if (finished_) {
            // 已经做完两轮 sweep：保持全局最优配置
            CpuFreqConfig best = bestConfigGlobal();
            setCurrentConfig(best.freqIdx, best.threadIdx);
            return;
        }

        // 若当前还有 sweep 队列没跑完，继续弹出
        if (!sweepQueue_.empty()) {
            CpuFreqConfig next = sweepQueue_.front();
            sweepQueue_.erase(sweepQueue_.begin());

            std::cout << "[LIN] queue pop cfg: freqIdx=" << next.freqIdx << " (" << freqLevels_[next.freqIdx]
                      << " kHz), threadIdx=" << next.threadIdx << " (n=" << threadLevels_[next.threadIdx] << ")\n";

            setCurrentConfig(next.freqIdx, next.threadIdx);
            return;
        }

        // 队列为空，说明当前维度的一次 sweep 跑完了，需要切换维度或收尾
        if (numSweepsDone_ >= 2) {
            // 2 个维度都扫完了：从整个 2D 空间中选最优
            CpuFreqConfig best = bestConfigGlobal();
            finished_          = true;

            std::cout << "[LIN] all sweeps done, global best cfg: freqIdx=" << best.freqIdx << " ("
                      << freqLevels_[best.freqIdx] << " kHz), threadIdx=" << best.threadIdx
                      << " (n=" << threadLevels_[best.threadIdx] << ")\n";

            setCurrentConfig(best.freqIdx, best.threadIdx);
            return;
        }

        // 还没完成全部维度，开始下一轮 sweep
        Dim nextDim = (lastSweep_ == Dim::THREAD ? Dim::FREQ : Dim::THREAD);
        ++numSweepsDone_;

        // 在上一次 sweep 的所有配置中，挑一个 cost 最小的作为基准
        CpuFreqConfig bestFromLast = lastSweepBackup_.front();
        double        bestCost     = costOf(bestFromLast.freqIdx, bestFromLast.threadIdx);

        for (const auto & cfg : lastSweepBackup_) {
            double c = costOf(cfg.freqIdx, cfg.threadIdx);
            if (!std::isinf(c) && c < bestCost) {
                bestCost     = c;
                bestFromLast = cfg;
            }
        }

        std::cout << "[LIN] base cfg for next sweep (dim=" << (nextDim == Dim::THREAD ? "THREAD" : "FREQ")
                  << "): freqIdx=" << bestFromLast.freqIdx << " (" << freqLevels_[bestFromLast.freqIdx]
                  << " kHz), threadIdx=" << bestFromLast.threadIdx << " (n=" << threadLevels_[bestFromLast.threadIdx]
                  << "), cost=" << bestCost << "\n";

        // 构造下一维度的 sweep 队列
        sweepQueue_.clear();
        lastSweepBackup_.clear();

        if (nextDim == Dim::THREAD) {
            // 固定频率，扫所有线程数
            int f = bestFromLast.freqIdx;
            for (int t = 0; t < (int) threadLevels_.size(); ++t) {
                CpuFreqConfig cfg{ f, t };
                sweepQueue_.push_back(cfg);
                lastSweepBackup_.push_back(cfg);
            }
        } else {  // Dim::FREQ
            // 固定线程数，扫所有频率
            int t = bestFromLast.threadIdx;
            for (int f = 0; f < (int) freqLevels_.size(); ++f) {
                CpuFreqConfig cfg{ f, t };
                sweepQueue_.push_back(cfg);
                lastSweepBackup_.push_back(cfg);
            }
        }

        lastSweep_ = nextDim;

        // 立刻弹出第一个，用于下一窗口
        CpuFreqConfig next = sweepQueue_.front();
        sweepQueue_.erase(sweepQueue_.begin());
        setCurrentConfig(next.freqIdx, next.threadIdx);
    }

  private:
    enum class Dim { THREAD = 0, FREQ = 1 };

    Dim                        lastSweep_;        // 上一次 sweep 的维度
    int                        numSweepsDone_;    // 已完成的 sweep 维度数（最多 2）
    bool                       finished_;         // 是否两维都扫完
    std::vector<CpuFreqConfig> sweepQueue_;       // 当前 sweep 的待测配置列表
    std::vector<CpuFreqConfig> lastSweepBackup_;  // 上一轮 sweep 的所有配置，用来选基准
};
