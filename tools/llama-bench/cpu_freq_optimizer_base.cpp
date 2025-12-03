#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// 一轮配置：频率档位 + 线程档位
struct CpuFreqConfig {
    int freqIdx;    // index into freqLevels_
    int threadIdx;  // index into threadLevels_
};

// 通用基类：负责
// - 维护 E/T cache
// - 计算代价 J
// - 存储每个 (freq, threads) 的历史 cost
// - 提供温度惩罚（可选）
class CpuFreqOptimizerBase {
  public:
    CpuFreqOptimizerBase(double                   alpha,
                         const std::vector<int> & freqLevels,    // 例如 kHz
                         const std::vector<int> & threadLevels,  // 例如 {1,2,4,6}
                         size_t                   cacheLength) :
        alpha_(alpha),
        freqLevels_(freqLevels),
        threadLevels_(threadLevels),
        cacheLength_(cacheLength),
        step_(0),
        energyBaseline_(std::numeric_limits<double>::quiet_NaN()),
        timeBaseline_(std::numeric_limits<double>::quiet_NaN()),
        cacheMaxTempC_(std::numeric_limits<double>::quiet_NaN()),
        currentCfg_{ 0, 0 },
        // 温度相关：默认关闭
        thermalEnabled_(false),
        thermalSoftC_(80.0),  // 惩罚开始的软阈值（示例：80℃
        thermalCritC_(90.0),  // 惩罚拉满的硬阈值（示例：90℃
        thermalWeight_(0.3),  // 惩罚权重
        rng_(std::random_device{}()) {
        if (freqLevels_.empty()) {
            throw std::runtime_error("freqLevels must not be empty");
        }
        if (threadLevels_.empty()) {
            throw std::runtime_error("threadLevels must not be empty");
        }

        const size_t nStates = freqLevels_.size() * threadLevels_.size();

        historyCost_.assign(nStates, std::numeric_limits<double>::quiet_NaN());
        historyStep_.assign(nStates, 0);

        // 默认从最高频 + 最大线程开始
        currentCfg_.freqIdx   = static_cast<int>(freqLevels_.size() - 1);
        currentCfg_.threadIdx = static_cast<int>(threadLevels_.size() - 1);
    }

    virtual ~CpuFreqOptimizerBase() = default;

    // ------------ 基本 getter ------------

    size_t cacheLength() const { return cacheLength_; }

    const CpuFreqConfig & currentConfig() const { return currentCfg_; }

    int currentFreqIndex() const { return currentCfg_.freqIdx; }

    int currentThreadIndex() const { return currentCfg_.threadIdx; }

    const std::vector<int> & freqLevels() const { return freqLevels_; }

    const std::vector<int> & threadLevels() const { return threadLevels_; }

    double alpha() const { return alpha_; }

    void set_alpha(double a) { alpha_ = a; }

    // 外部可以手工设 baseline；否则会自动估计
    void setBaseline(double e0, double t0) {
        energyBaseline_ = e0;
        timeBaseline_   = t0;
        std::cout << "[OPT] Baseline manually set: E0=" << energyBaseline_ << " T0=" << timeBaseline_ << std::endl;
    }

    // ------------ 温度惩罚相关接口 ------------

    // 开/关温度惩罚（默认关闭）
    void set_thermal_enabled(bool enabled) { thermalEnabled_ = enabled; }

    bool thermal_enabled() const { return thermalEnabled_; }

    // 设置温度参数：软阈值、硬阈值（℃）、权重
    void set_thermal_params(double softC, double critC, double weight) {
        thermalSoftC_  = softC;
        thermalCritC_  = critC;
        thermalWeight_ = weight;
    }

    // ------------ 窗口结束：喂入样本（带温度版本） ------------

    // energies / latencies：本窗口内所有请求的样本
    // windowMaxTempC：本窗口内观测到的最大温度（℃），如果没有温度就传 NaN
    void postBatch(const std::vector<double> & energies, const std::vector<double> & latencies, double windowMaxTempC) {
        if (energies.size() != latencies.size()) {
            throw std::runtime_error("energies and latencies size mismatch");
        }

        cacheEnergy_.insert(cacheEnergy_.end(), energies.begin(), energies.end());
        cacheTime_.insert(cacheTime_.end(), latencies.begin(), latencies.end());

        // 记录当前窗口内的最大温度
        if (std::isnan(cacheMaxTempC_)) {
            cacheMaxTempC_ = windowMaxTempC;
        } else if (!std::isnan(windowMaxTempC)) {
            cacheMaxTempC_ = std::max(cacheMaxTempC_, windowMaxTempC);
        }

        // 样本够一个 window 了（比如 10 次请求）
        if (cacheEnergy_.size() >= cacheLength_) {
            std::cout << "[OPT] ---- step " << step_ << " update begin, freqIdx=" << currentCfg_.freqIdx << " ("
                      << freqLevels_[currentCfg_.freqIdx] << " kHz)"
                      << ", threadIdx=" << currentCfg_.threadIdx << " (n=" << threadLevels_[currentCfg_.threadIdx]
                      << ")"
                      << ", samples=" << cacheEnergy_.size() << ", maxTemp=" << cacheMaxTempC_ << " C ----"
                      << std::endl;

            updateHistory(cacheMaxTempC_);

            CpuFreqConfig best     = bestConfigGlobal();
            double        bestCost = costOf(best.freqIdx, best.threadIdx);

            std::cout << "[OPT] after step " << step_ << " best=(freqIdx=" << best.freqIdx << ", "
                      << freqLevels_[best.freqIdx] << " kHz"
                      << "; threadIdx=" << best.threadIdx << ", n=" << threadLevels_[best.threadIdx] << ")"
                      << " bestCost=" << bestCost << std::endl;

            // 子类决定下一轮配置
            chooseNextConfig();

            ++step_;
        }
    }

    // 兼容旧代码：不带温度（相当于 windowMaxTempC=NaN → 无温度惩罚）
    void postBatch(const std::vector<double> & energies, const std::vector<double> & latencies) {
        postBatch(energies, latencies, std::numeric_limits<double>::quiet_NaN());
    }

  protected:
    // 子类需要实现：决定下一次用哪个 (freq, threads)
    virtual void chooseNextConfig() = 0;

    // 默认的 history 更新：支持温度惩罚
    virtual void updateHistory(double windowMaxTempC) {
        if (cacheEnergy_.empty()) {
            std::cout << "[OPT] updateHistory: empty cache, skip" << std::endl;
            return;
        }

        const double meanE = mean(cacheEnergy_.begin(), cacheEnergy_.end());
        const double meanT = mean(cacheTime_.begin(), cacheTime_.end());

        // 如果还没有 baseline，就用当前 step 的均值初始化一次
        if (std::isnan(energyBaseline_) || std::isnan(timeBaseline_)) {
            energyBaseline_ = meanE;
            timeBaseline_   = meanT;
            std::cout << "[OPT] Baseline auto-estimated at step " << step_ << " from (freqIdx=" << currentCfg_.freqIdx
                      << ", " << freqLevels_[currentCfg_.freqIdx] << " kHz; "
                      << "threadIdx=" << currentCfg_.threadIdx << ", n=" << threadLevels_[currentCfg_.threadIdx] << ")"
                      << " -> E0=" << energyBaseline_ << " T0=" << timeBaseline_ << std::endl;
        }

        // 原始 E/T cost
        double baseCost = alpha_ * (meanE / energyBaseline_) + (1.0 - alpha_) * (meanT / timeBaseline_);

        // 温度惩罚：接近阈值才生效
        double penaltyFactor = thermalPenaltyFactor(windowMaxTempC);
        double totalCost     = baseCost;

        if (thermalEnabled_) {
            totalCost = baseCost * (1.0 + thermalWeight_ * penaltyFactor);
        }

        const int fIdx   = currentCfg_.freqIdx;
        const int tIdx   = currentCfg_.threadIdx;
        const int flatId = flatIndex(fIdx, tIdx);

        historyCost_[flatId] = totalCost;
        historyStep_[flatId] = step_;

        std::cout << "[OPT] step " << step_ << " (freqIdx=" << fIdx << ", " << freqLevels_[fIdx] << " kHz"
                  << "; threadIdx=" << tIdx << ", n=" << threadLevels_[tIdx] << ")"
                  << " meanE=" << meanE << " meanT=" << meanT << " baseCost=" << baseCost
                  << " maxTemp=" << windowMaxTempC << " penaltyFactor=" << penaltyFactor
                  << " thermalEnabled=" << thermalEnabled_ << " thermalWeight=" << thermalWeight_
                  << " totalCost=" << totalCost << std::endl;
        // ★★★★★ 调用子类 hook
        onNewCost(currentCfg_.freqIdx, currentCfg_.threadIdx, totalCost);
        cacheEnergy_.clear();
        cacheTime_.clear();
        cacheMaxTempC_ = std::numeric_limits<double>::quiet_NaN();
    }

    // 新增：默认空实现，子类可以根据需要 override
    virtual void onNewCost(int /*fIdx*/, int /*tIdx*/, double /*effCost*/) {
        // 默认啥也不干
    }

    // 兼容老签名：不带温度
    virtual void updateHistory() { updateHistory(std::numeric_limits<double>::quiet_NaN()); }

    template <typename It> static double mean(It first, It last) {
        const auto n = std::distance(first, last);
        if (n <= 0) {
            return 0.0;
        }
        double sum = std::accumulate(first, last, 0.0);
        return sum / static_cast<double>(n);
    }

    bool validIdx(int fIdx, int tIdx) const {
        return fIdx >= 0 && fIdx < (int) freqLevels_.size() && tIdx >= 0 && tIdx < (int) threadLevels_.size();
    }

    int flatIndex(int fIdx, int tIdx) const { return fIdx * (int) threadLevels_.size() + tIdx; }

    double costOf(int fIdx, int tIdx) const {
        if (!validIdx(fIdx, tIdx)) {
            return std::numeric_limits<double>::infinity();
        }
        double c = historyCost_[flatIndex(fIdx, tIdx)];
        if (std::isnan(c)) {
            return std::numeric_limits<double>::infinity();
        }
        return c;
    }

    // 返回整个 2D 空间里 cost 最小的配置（只看已有观测）
    CpuFreqConfig bestConfigGlobal() const {
        double        bestCost = std::numeric_limits<double>::infinity();
        CpuFreqConfig bestCfg{ 0, 0 };

        for (int f = 0; f < (int) freqLevels_.size(); ++f) {
            for (int t = 0; t < (int) threadLevels_.size(); ++t) {
                double c = historyCost_[flatIndex(f, t)];
                if (!std::isnan(c) && c < bestCost) {
                    bestCost          = c;
                    bestCfg.freqIdx   = f;
                    bestCfg.threadIdx = t;
                }
            }
        }
        return bestCfg;
    }

    void setCurrentConfig(int fIdx, int tIdx) {
        if (!validIdx(fIdx, tIdx)) {
            throw std::runtime_error("invalid freq or thread idx");
        }
        currentCfg_.freqIdx   = fIdx;
        currentCfg_.threadIdx = tIdx;
    }

    // 温度惩罚：返回 [0,1] 的强度，0 表示无惩罚
    double thermalPenaltyFactor(double maxTempC) const {
        if (!thermalEnabled_) {
            return 0.0;
        }

        if (std::isnan(maxTempC)) {
            return 0.0;  // 没有温度信息，不惩罚
        }

        if (maxTempC <= thermalSoftC_) {
            return 0.0;  // 低于软阈值，无惩罚
        }

        if (maxTempC >= thermalCritC_) {
            return 1.0;  // 超过硬阈值，惩罚拉满
        }

        // 映射到 (0, 1) 区间
        double x     = (maxTempC - thermalSoftC_) / (thermalCritC_ - thermalSoftC_);
        // gamma>1：刚过软阈值时惩罚缓，越接近 T_crit 越陡
        double gamma = 2.0;
        return std::pow(x, gamma);
    }

  protected:
    // -------- 代价函数参数 --------
    double alpha_;

    // -------- 配置空间 --------
    std::vector<int> freqLevels_;    // 频率列表（kHz）
    std::vector<int> threadLevels_;  // 线程数列表（例如 {1,2,4,...}）
    size_t           cacheLength_;   // 每个 window 内的样本数
    size_t           step_;          // 第几次 window

    // -------- baseline --------
    double energyBaseline_;
    double timeBaseline_;

    // -------- 当前 window 的原始样本缓存 --------
    std::vector<double> cacheEnergy_;
    std::vector<double> cacheTime_;
    double              cacheMaxTempC_;

    // -------- 历史 cost --------
    // 2D (freq, threads) 压成 1D 存
    std::vector<double> historyCost_;
    std::vector<size_t> historyStep_;

    // -------- 当前配置 --------
    CpuFreqConfig        currentCfg_;
    mutable std::mt19937 rng_;

    // -------- 温度惩罚参数 --------
    bool   thermalEnabled_;  // 是否启用温度惩罚
    double thermalSoftC_;    // 惩罚开始的软阈值
    double thermalCritC_;    // 惩罚拉满的硬阈值
    double thermalWeight_;   // 温度惩罚在总 cost 中的权重
};
