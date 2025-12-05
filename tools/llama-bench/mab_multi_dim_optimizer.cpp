// mab_multi_dim_optimizer.h
#pragma once

#include "cpu_freq_optimizer_base.cpp"

#include <cmath>
#include <limits>
#include <random>

// 多维 MAB 优化器：对应 python 里的 EnergyOptimizer_MAB_multiDim，
// 只不过这里是 2 维 (freqIdx, threadIdx)，没有 GPU / batch 那一维。
class MABMultiDimCpuFreqOptimizer : public CpuFreqOptimizerBase {
  public:
    // orderDims 例如 {Dim::FREQ, Dim::THREAD} 或 {Dim::THREAD, Dim::FREQ}
    // expAvgAlpha: 指数滑动平均 EMA 的 alpha（越大越偏向最近一次）
    // exploitProb: 利用（exploit）的概率，1-exploitProb 为随机探索
    // hotStart: 前 hotStart 个窗口用随机配置做“热身”
    MABMultiDimCpuFreqOptimizer(double                   alpha,
                                const std::vector<int> & freqLevels,
                                const std::vector<int> & threadLevels,
                                size_t                   cacheLength,
                                std::vector<int>         freqMinMaxIdx   = {},  // 可选：限制频率索引范围
                                std::vector<int>         threadMinMaxIdx = {},
                                double                   expAvgAlpha     = 0.9,
                                double                   exploitProb     = 0.9,
                                int                      hotStart        = 10,
                                std::vector<int>         orderDims       = {}) :
        CpuFreqOptimizerBase(alpha, freqLevels, threadLevels, cacheLength),
        expAvgAlpha_(expAvgAlpha),
        exploitProb_(exploitProb),
        hotStart_(std::min(static_cast<int>(cacheLength), hotStart)),  // 取最小值
        orderOffsetInitialized_(false) {
        // 维度顺序：默认 [FREQ, THREAD]
        if (orderDims.empty()) {
            order_ = { Dim::FREQ, Dim::THREAD };
        } else {
            order_.clear();
            for (int d : orderDims) {
                order_.push_back(d == 0 ? Dim::FREQ : Dim::THREAD);
            }
        }

        // 最小/最大 index 限制（不传就用全范围）
        if (freqMinMaxIdx.size() == 2) {
            freqMinIdx_ = std::max(0, freqMinMaxIdx[0]);
            freqMaxIdx_ = std::min((int) freqLevels_.size() - 1, freqMinMaxIdx[1]);
        } else {
            freqMinIdx_ = 0;
            freqMaxIdx_ = (int) freqLevels_.size() - 1;
        }

        if (threadMinMaxIdx.size() == 2) {
            threadMinIdx_ = std::max(0, threadMinMaxIdx[0]);
            threadMaxIdx_ = std::min((int) threadLevels_.size() - 1, threadMinMaxIdx[1]);
        } else {
            threadMinIdx_ = 0;
            threadMaxIdx_ = (int) threadLevels_.size() - 1;
        }

        std::cout << "[MAB] init: expAvgAlpha=" << expAvgAlpha_ << " exploitProb=" << exploitProb_
                  << " hotStart=" << hotStart_ << " freqIdx in [" << freqMinIdx_ << "," << freqMaxIdx_
                  << "], threadIdx in [" << threadMinIdx_ << "," << threadMaxIdx_ << "]\n";
    }

  protected:
    // ===== 重写：用 EMA 更新 historyCost_，而不是简单覆盖 =====

    void onNewCost(int fIdx, int tIdx, double effCost) override {
        int      id   = flatIndex(fIdx, tIdx);
        double & slot = historyCost_[id];

        if (std::isnan(slot)) {
            slot = effCost;
        } else {
            slot = (1.0 - expAvgAlpha_) * slot + expAvgAlpha_ * effCost;
        }
    }

    // ===== 重写：每个 window 结束时选择下一轮真实 window 的配置 =====
    void chooseNextConfig() override {
        // 1. hot-start：先随机探索若干轮
        if (hotStart_ > 0) {
            --hotStart_;
            CpuFreqConfig cfg = randomConfig();
            std::cout << "[MAB] hot-start, remaining=" << hotStart_ << " next cfg=(fIdx=" << cfg.freqIdx
                      << ", tIdx=" << cfg.threadIdx << ")\n";
            setCurrentConfig(cfg.freqIdx, cfg.threadIdx);
            return;
        }

        // 2. 确定当前要优化哪个维度（Freq or Thread）
        if (!orderOffsetInitialized_) {
            // 对齐到当前 step_，类似 python 里的 order_offset
            orderOffset_            = step_ % order_.size();
            orderOffsetInitialized_ = true;
        }

        Dim dimToSet = order_[(step_ - orderOffset_) % order_.size()];

        std::uniform_real_distribution<double> uni(0.0, 1.0);
        const double                           r = uni(rng_);

        if (dimToSet == Dim::FREQ) {
            // ===== 先优化频率维度：对每个 freqIdx 取所有 threadIdx 上的 nanmean =====
            int    fixedT  = currentCfg_.threadIdx;
            int    newF    = currentCfg_.freqIdx;
            double bestAvg = std::numeric_limits<double>::infinity();

            if (r < exploitProb_) {
                // exploit：找频率维度的“最好档”
                for (int f = freqMinIdx_; f <= freqMaxIdx_; ++f) {
                    double avg = nanMeanOverThreads(f);
                    if (std::isnan(avg)) {
                        continue;  // 这个频点完全没数据，先跳过
                    }
                    if (avg < bestAvg) {
                        bestAvg = avg;
                        newF    = f;
                    }
                }

                // 如果全部都是 NaN，就退化成随机
                if (std::isinf(bestAvg)) {
                    newF = randomFreqIdx();
                }

                std::cout << "[MAB] FREQ exploit: choose fIdx=" << newF << " (kHz=" << freqLevels_[newF]
                          << "), fixed tIdx=" << fixedT << " bestAvgCost=" << bestAvg << "\n";
            } else {
                // explore：随机选一个 freqIdx
                newF = randomFreqIdx();
                std::cout << "[MAB] FREQ explore: random fIdx=" << newF << " (kHz=" << freqLevels_[newF]
                          << "), fixed tIdx=" << fixedT << "\n";
            }

            setCurrentConfig(newF, fixedT);
        } else {
            // ===== 先优化线程维度：对每个 threadIdx 取所有 freqIdx 上的 nanmean =====
            int    fixedF  = currentCfg_.freqIdx;
            int    newT    = currentCfg_.threadIdx;
            double bestAvg = std::numeric_limits<double>::infinity();

            if (r < exploitProb_) {
                for (int t = threadMinIdx_; t <= threadMaxIdx_; ++t) {
                    double avg = nanMeanOverFreqs(t);
                    if (std::isnan(avg)) {
                        continue;
                    }
                    if (avg < bestAvg) {
                        bestAvg = avg;
                        newT    = t;
                    }
                }

                if (std::isinf(bestAvg)) {
                    newT = randomThreadIdx();
                }

                std::cout << "[MAB] THREAD exploit: choose tIdx=" << newT << " (n=" << threadLevels_[newT]
                          << "), fixed fIdx=" << fixedF << " bestAvgCost=" << bestAvg << "\n";
            } else {
                newT = randomThreadIdx();
                std::cout << "[MAB] THREAD explore: random tIdx=" << newT << " (n=" << threadLevels_[newT]
                          << "), fixed fIdx=" << fixedF << "\n";
            }

            setCurrentConfig(fixedF, newT);
        }
    }

  private:
    enum class Dim { FREQ = 0, THREAD = 1 };

    // 在某一频点上，对所有线程取 nanmean(cost)
    double nanMeanOverThreads(int fIdx) const {
        double sum = 0.0;
        int    cnt = 0;
        for (int t = threadMinIdx_; t <= threadMaxIdx_; ++t) {
            double c = historyCost_[flatIndex(fIdx, t)];
            if (!std::isnan(c)) {
                sum += c;
                ++cnt;
            }
        }
        if (cnt == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return sum / (double) cnt;
    }

    // 在某一线程数上，对所有频点取 nanmean(cost)
    double nanMeanOverFreqs(int tIdx) const {
        double sum = 0.0;
        int    cnt = 0;
        for (int f = freqMinIdx_; f <= freqMaxIdx_; ++f) {
            double c = historyCost_[flatIndex(f, tIdx)];
            if (!std::isnan(c)) {
                sum += c;
                ++cnt;
            }
        }
        if (cnt == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return sum / (double) cnt;
    }

    int randomFreqIdx() {
        std::uniform_int_distribution<int> dist(freqMinIdx_, freqMaxIdx_);
        return dist(rng_);
    }

    int randomThreadIdx() {
        std::uniform_int_distribution<int> dist(threadMinIdx_, threadMaxIdx_);
        return dist(rng_);
    }

    CpuFreqConfig randomConfig() { return CpuFreqConfig{ randomFreqIdx(), randomThreadIdx() }; }

  private:
    // 参数
    double           expAvgAlpha_;   // EMA alpha
    double           exploitProb_;   // 利用概率
    int              hotStart_;      // 热身窗口数（随机配置）
    int              freqMinIdx_;    // 频率索引下限
    int              freqMaxIdx_;    // 频率索引上限
    int              threadMinIdx_;  // 线程索引下限
    int              threadMaxIdx_;  // 线程索引上限
    std::vector<Dim> order_;         // 维度优化顺序（默认 FREQ -> THREAD）

    // 对齐 step_ 的 offset
    size_t orderOffset_;
    bool   orderOffsetInitialized_;
};
