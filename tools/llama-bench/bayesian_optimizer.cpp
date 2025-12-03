#pragma once

#include "cpu_freq_optimizer_base.cpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// 基于 GP + Expected Improvement 的贝叶斯优化器
// 域：freqLevels_ × threadLevels_
// cost 从基类 historyCost_ 中取（J = alpha * E_norm + (1-alpha) * T_norm）
// y = -cost 作为 GP 回归目标（相当于最大化 -cost，等价于最小化 cost）
class BayesianCpuFreqOptimizer : public CpuFreqOptimizerBase {
  public:
    // hotStart: 先随机挑一些未探索配置做“冷启动”轮次
    BayesianCpuFreqOptimizer(double                   alpha,
                             const std::vector<int> & freqLevels,
                             const std::vector<int> & threadLevels,
                             size_t                   cacheLength,
                             int                      hotStart = 5) :
        CpuFreqOptimizerBase(alpha, freqLevels, threadLevels, cacheLength),
        hotStart_(hotStart) {}

  protected:
    void chooseNextConfig() override {
        // 1. 收集所有已经有真实 cost 的点，构建 GP 训练数据
        std::vector<std::array<double, 2>> X;  // (freqIdx, threadIdx)
        std::vector<double>                y;  // 目标：-cost（越大越好）
        double                             best_y = -std::numeric_limits<double>::infinity();

        for (int f = 0; f < (int) freqLevels_.size(); ++f) {
            for (int t = 0; t < (int) threadLevels_.size(); ++t) {
                double c = costOf(f, t);  // 基类：NaN/未测返回 +inf
                if (std::isnan(c) || std::isinf(c)) {
                    continue;
                }
                double yy = -c;
                X.push_back({ (double) f, (double) t });
                y.push_back(yy);
                if (yy > best_y) {
                    best_y = yy;
                }
            }
        }

        // 2. hot-start：先随机测一些“未探索配置”
        if (hotStart_ > 0) {
            std::vector<CpuFreqConfig> unobserved;
            for (int f = 0; f < (int) freqLevels_.size(); ++f) {
                for (int t = 0; t < (int) threadLevels_.size(); ++t) {
                    double c = costOf(f, t);
                    if (std::isinf(c) || std::isnan(c)) {
                        unobserved.push_back({ f, t });
                    }
                }
            }

            if (!unobserved.empty()) {
                std::uniform_int_distribution<size_t> dist(0, unobserved.size() - 1);
                CpuFreqConfig                         pick = unobserved[dist(rng_)];

                std::cout << "[BO] hot-start pick cfg=(freqIdx=" << pick.freqIdx << " (" << freqLevels_[pick.freqIdx]
                          << " kHz)"
                          << ", threadIdx=" << pick.threadIdx << " (n=" << threadLevels_[pick.threadIdx] << "))\n";

                setCurrentConfig(pick.freqIdx, pick.threadIdx);
                --hotStart_;
                return;
            } else {
                hotStart_ = 0;
            }
        }

        // 3. 训练数据太少：退化为“选历史最优配置”
        if (X.size() < 2 || !std::isfinite(best_y)) {
            CpuFreqConfig best = bestConfigGlobal();
            setCurrentConfig(best.freqIdx, best.threadIdx);
            std::cout << "[BO] not enough data, fallback to global best cfg=("
                      << "freqIdx=" << best.freqIdx << ", threadIdx=" << best.threadIdx << ")\n";
            return;
        }

        // 4. 拟合极简 GP
        GPModel gp;
        gp.lengthScale = 1.0;
        gp.noise       = 1e-3;
        gp.X           = std::move(X);
        gp.y           = std::move(y);
        gp.fit();

        // 5. 在整个 freq×thread 空间上计算 Expected Improvement，选择 EI 最大的配置
        double        bestEI  = -std::numeric_limits<double>::infinity();
        CpuFreqConfig bestCfg = currentConfig();

        for (int f = 0; f < (int) freqLevels_.size(); ++f) {
            for (int t = 0; t < (int) threadLevels_.size(); ++t) {
                std::array<double, 2> xStar{ (double) f, (double) t };
                double                mu, sigma;
                gp.predict(xStar, mu, sigma);

                if (sigma < 1e-9) {
                    continue;  // 没不确定性，EI≈0
                }

                double z  = (mu - best_y) / sigma;
                double ei = expectedImprovement(mu, sigma, best_y, z);
                if (ei > bestEI) {
                    bestEI  = ei;
                    bestCfg = { f, t };
                }
            }
        }

        std::cout << "[BO] next cfg from EI: freqIdx=" << bestCfg.freqIdx << " (" << freqLevels_[bestCfg.freqIdx]
                  << " kHz)"
                  << ", threadIdx=" << bestCfg.threadIdx << " (n=" << threadLevels_[bestCfg.threadIdx] << ")"
                  << ", bestEI=" << bestEI << "\n";

        setCurrentConfig(bestCfg.freqIdx, bestCfg.threadIdx);
    }

  private:
    // 极简 Gaussian Process 模型（RBF kernel + 噪声）
    struct GPModel {
        std::vector<std::array<double, 2>> X;  // 输入：freqIdx, threadIdx
        std::vector<double>                y;  // 输出：-cost
        double                             lengthScale = 1.0;
        double                             noise       = 1e-3;
        std::vector<std::vector<double>>   Kinv;  // 协方差矩阵的逆

        // RBF kernel: k(x,x') = exp(-0.5 * ||x-x'||^2 / ℓ^2) + σ_n^2 δ_ij
        void fit() {
            const size_t n = X.size();
            if (n == 0) {
                return;
            }

            std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));
            const double                     ls2 = lengthScale * lengthScale;

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double d0 = X[i][0] - X[j][0];
                    double d1 = X[i][1] - X[j][1];
                    double r2 = d0 * d0 + d1 * d1;
                    double k  = std::exp(-0.5 * r2 / ls2);
                    if (i == j) {
                        k += noise * noise;
                    }
                    K[i][j] = k;
                }
            }

            Kinv = invertMatrix(K);
        }

        // 预测：给一个新点 x*，输出 GP 均值和标准差
        void predict(const std::array<double, 2> & xStar, double & mean, double & stddev) const {
            const size_t n = X.size();
            if (n == 0 || Kinv.empty()) {
                mean   = 0.0;
                stddev = 1.0;
                return;
            }

            const double        ls2 = lengthScale * lengthScale;
            std::vector<double> kvec(n);
            for (size_t i = 0; i < n; ++i) {
                double d0 = X[i][0] - xStar[0];
                double d1 = X[i][1] - xStar[1];
                double r2 = d0 * d0 + d1 * d1;
                kvec[i]   = std::exp(-0.5 * r2 / ls2);
            }
            double kStar = 1.0;  // k(x*,x*)

            // alpha = Kinv * y
            std::vector<double> alpha(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    alpha[i] += Kinv[i][j] * y[j];
                }
            }

            // mean = k^T * alpha
            mean = 0.0;
            for (size_t i = 0; i < n; ++i) {
                mean += kvec[i] * alpha[i];
            }

            // v = Kinv * k
            std::vector<double> v(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    v[i] += Kinv[i][j] * kvec[j];
                }
            }

            double kTKinvk = 0.0;
            for (size_t i = 0; i < n; ++i) {
                kTKinvk += kvec[i] * v[i];
            }

            double var = kStar - kTKinvk;
            if (var < 1e-12) {
                var = 1e-12;
            }
            stddev = std::sqrt(var);
        }

        // 简单 Gauss-Jordan 反矩阵
        static std::vector<std::vector<double>> invertMatrix(const std::vector<std::vector<double>> & A) {
            const size_t                     n = A.size();
            std::vector<std::vector<double>> B(n, std::vector<double>(2 * n, 0.0));

            // [A | I]
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    B[i][j] = A[i][j];
                }
                B[i][n + i] = 1.0;
            }

            // Gauss-Jordan 消元
            for (size_t i = 0; i < n; ++i) {
                double pivot = B[i][i];
                if (std::fabs(pivot) < 1e-12) {
                    // 简单 pivot 修正：找下面一行更大的
                    for (size_t r = i + 1; r < n; ++r) {
                        if (std::fabs(B[r][i]) > std::fabs(pivot)) {
                            std::swap(B[i], B[r]);
                            pivot = B[i][i];
                            break;
                        }
                    }
                }
                if (std::fabs(pivot) < 1e-12) {
                    continue;
                }
                double invPivot = 1.0 / pivot;
                for (size_t j = 0; j < 2 * n; ++j) {
                    B[i][j] *= invPivot;
                }
                for (size_t r = 0; r < n; ++r) {
                    if (r == i) {
                        continue;
                    }
                    double factor = B[r][i];
                    if (factor == 0.0) {
                        continue;
                    }
                    for (size_t c = 0; c < 2 * n; ++c) {
                        B[r][c] -= factor * B[i][c];
                    }
                }
            }

            // 提取右半部分作为 A^{-1}
            std::vector<std::vector<double>> Inv(n, std::vector<double>(n, 0.0));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    Inv[i][j] = B[i][n + j];
                }
            }
            return Inv;
        }
    };

    static double normalPdf(double z) {
        static const double invSqrt2Pi = 0.39894228040143267794;  // 1/sqrt(2π)
        return invSqrt2Pi * std::exp(-0.5 * z * z);
    }

    static double normalCdf(double z) { return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0))); }

    // EI 公式：针对“max y”场景（这里 y = -cost）
    static double expectedImprovement(double mu, double sigma, double best_y, double z) {
        double Phi = normalCdf(z);
        double phi = normalPdf(z);
        double ei  = (mu - best_y) * Phi + sigma * phi;
        if (ei < 0.0) {
            ei = 0.0;
        }
        return ei;
    }

  private:
    int hotStart_;
};
