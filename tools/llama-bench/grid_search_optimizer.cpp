#pragma once

#include "cpu_freq_optimizer_base.cpp"

#include <iostream>
#include <vector>

// 2D 网格搜索：遍历所有 (freqIdx, threadIdx) 组合，最后固定在全局最优
// 语义对齐 EnergyOptimizer_GridSearch：
// - 用真实请求一个一个跑完所有档位
// - 全部测完后，从 historyCost_ 里找最优配置
// - 后续一直使用该配置（baseline，不再动态调整）
class GridSearchCpuFreqOptimizer : public CpuFreqOptimizerBase {
  public:
    GridSearchCpuFreqOptimizer(double                   alpha,
                               const std::vector<int> & freqLevels,
                               const std::vector<int> & threadLevels,
                               size_t                   cacheLength) :
        CpuFreqOptimizerBase(alpha, freqLevels, threadLevels, cacheLength),
        gridInitialized_(false),
        gridFinished_(false) {
        // 起点沿用基类的“最高频 + 最大线程”
        // 如需从别的点开始，可以在外部 setCurrentConfig(...)
    }

  protected:
    void chooseNextConfig() override {
        // 如果还没初始化网格队列：构造一次完整 sweep 队列
        if (!gridInitialized_) {
            const int nFreq   = static_cast<int>(freqLevels_.size());
            const int nThread = static_cast<int>(threadLevels_.size());

            CpuFreqConfig cur = currentConfig();

            for (int f = 0; f < nFreq; ++f) {
                for (int t = 0; t < nThread; ++t) {
                    // 起点这格已经在当前窗口测过一次，可以选择跳过避免重复；也可以保留
                    if (f == cur.freqIdx && t == cur.threadIdx) {
                        continue;
                    }
                    sweepQueue_.push_back({ f, t });
                }
            }
            gridInitialized_ = true;

            std::cout << "[GRID] initialized sweep queue, size=" << sweepQueue_.size() << std::endl;
        }

        // 1）如果网格还没扫完：弹出队列头一个配置
        if (!sweepQueue_.empty()) {
            CpuFreqConfig next = sweepQueue_.front();
            sweepQueue_.erase(sweepQueue_.begin());

            std::cout << "[GRID] next sweep cfg: freqIdx=" << next.freqIdx << " (" << freqLevels_[next.freqIdx]
                      << " kHz), threadIdx=" << next.threadIdx << " (n=" << threadLevels_[next.threadIdx] << ")"
                      << std::endl;

            setCurrentConfig(next.freqIdx, next.threadIdx);
            return;
        }

        // 2）队列空了，说明所有组合都测完一次
        if (!gridFinished_) {
            CpuFreqConfig best = bestConfigGlobal();
            gridFinished_      = true;

            std::cout << "[GRID] sweep finished, best cfg: freqIdx=" << best.freqIdx << " ("
                      << freqLevels_[best.freqIdx] << " kHz), threadIdx=" << best.threadIdx
                      << " (n=" << threadLevels_[best.threadIdx] << ")" << std::endl;

            // 之后一直用这个全局最优配置
            setCurrentConfig(best.freqIdx, best.threadIdx);
            return;
        }

        // 3）gridFinished_ == true：已经选出最优，后续每次都保持该配置
        {
            CpuFreqConfig best = bestConfigGlobal();
            setCurrentConfig(best.freqIdx, best.threadIdx);
        }
    }

  private:
    std::vector<CpuFreqConfig> sweepQueue_;
    bool                       gridInitialized_;
    bool                       gridFinished_;
};
