// dynamic_opt_main.cpp

#include "bayesian_optimizer.cpp"
#include "cpu_freq_optimizer_base.cpp"
#include "ggml.h"
#include "grid_search_optimizer.cpp"
#include "linear_search_optimizer.cpp"
#include "llama.h"
#include "llama_runner.cpp"
#include "mab_multi_dim_optimizer.cpp"
#include "neighbor_search_optimizer.cpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ============== 简单 helper：算平均值 ==============

static double mean(const std::vector<double> & v) {
    if (v.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / static_cast<double>(v.size());
}

// ============== 占位：实际设置 CPU 频点 ==============
// TODO: 替换成你真机上的 DVFS 控制逻辑，比如写 /sys/devices/system/cpu/...
static void apply_cpu_freq_khz(int freq_khz) {
    std::cout << "[DVFS] set CPU freq to " << freq_khz << " kHz" << std::endl;
}

// ============== 占位：本窗口最大温度（℃） ==============
// 现在默认返回 NaN，相当于“没有温度信息 → 不启用温度惩罚”。
// 以后你可以改成：在一个 window 内定期读 /sys/class/thermal/...，取 max。
static double read_window_max_temp_c() {
    return std::numeric_limits<double>::quiet_NaN();
}

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

// 选择当前想测试的优化器类型：Grid / Linear / Neighbor / Bayesian / MAB
// using OptimizerT = GridSearchCpuFreqOptimizer;
// using OptimizerT = LinearSearchOptimizer;
// using OptimizerT = NeighborSearchOptimizer;
// using OptimizerT = BayesianCpuFreqOptimizer;
using OptimizerT = MABMultiDimCpuFreqOptimizer;

// ============== 主流程：只有“真实窗口”，每窗口结束调一次优化器 ==============

int main(int argc, char ** argv) {
    llama_log_set(llama_null_log_callback, nullptr);

    // -------- 1. 基础初始化（一次性） --------
    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // 模型路径，可通过命令行传入
    std::string model_path = "/data/local/tmp/cpp/Qwen3-0.6B-Q4_0.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    // -------- 2. 创建 LlamaRunner --------
    const int n_ctx        = 512;  // 至少 >= prompt + gen
    const int n_batch      = 2048;
    const int init_threads = 4;    // 初始线程数，后面会被动态修改

    LlamaRunner runner(model_path, n_ctx, n_batch, init_threads);

    // -------- 3. 定义可选的 CPU 频率档位（kHz） --------
    std::vector<int> freqLevelsKHz = {
        844800, 1190400, 1497600, 1785600, 2073600, 2352000,
    };

    // -------- 3.1 定义可选的线程数档位 --------
    std::vector<int> threadLevels = {
        1,
        2,
        4,
    };

    // 每一“窗口”用多少次推理样本进行统计
    const size_t SAMPLES_PER_WINDOW = 10;

    // 单次请求的 token 规模（你可以按场景改）
    const int PROMPT_TOKENS = 64;
    const int GEN_TOKENS    = 32;
    const int TOTAL_TOKENS  = PROMPT_TOKENS + GEN_TOKENS;

    // 代价函数权重 alpha：J = alpha * E_norm + (1-alpha) * T_norm
    const double alpha = 0.5;

    // -------- 4. 创建优化器 --------
    OptimizerT optimizer(alpha, freqLevelsKHz, threadLevels, SAMPLES_PER_WINDOW);

    // 如需启用温度惩罚，在这里打开
    // optimizer.set_thermal_enabled(true);
    // optimizer.set_thermal_params(/*softC=*/80.0, /*critC=*/90.0, /*weight=*/0.3);

    // -------- 5. 主循环：真实请求 + 每窗口调一次优化器 --------
    const int MAX_WINDOWS = 100;  // 真实窗口总数（实验上限）

    for (int w = 0; w < MAX_WINDOWS; ++w) {
        // 当前窗口配置：由优化器给出
        CpuFreqConfig cfg = optimizer.currentConfig();

        int curFreqKHz = freqLevelsKHz[cfg.freqIdx];
        int curThreads = threadLevels[cfg.threadIdx];

        std::cout << "\n==================== WINDOW " << w << " : freqIdx=" << cfg.freqIdx << " (" << curFreqKHz
                  << " kHz)"
                  << ", threadIdx=" << cfg.threadIdx << " (n=" << curThreads << ")"
                  << " ====================\n";

        // 下发 DVFS 配置 + 线程配置
        apply_cpu_freq_khz(curFreqKHz);
        runner.set_freq_ghz(curFreqKHz / 1e6);  // kHz -> GHz（供伪能耗模型使用）
        runner.set_num_threads(curThreads);

        // 在当前配置下，跑 SAMPLES_PER_WINDOW 次 PROMPT_TOKENS + GEN_TOKENS 推理
        std::vector<double> energies_per_token;
        std::vector<double> latencies_per_token;  // 用 steady per-token latency（秒/Token）

        energies_per_token.reserve(SAMPLES_PER_WINDOW);
        latencies_per_token.reserve(SAMPLES_PER_WINDOW);

        for (size_t i = 0; i < SAMPLES_PER_WINDOW; ++i) {
            auto m = runner.run_one_request(/*n_prompt=*/PROMPT_TOKENS, /*n_gen=*/GEN_TOKENS);

            if (m.steady_ts <= 0.0) {
                std::cerr << "[WIN " << w << "] warning: steady_ts <= 0, skip sample " << i << " cfg=(" << curFreqKHz
                          << " kHz, n=" << curThreads << ")\n";
                continue;
            }

            // ==== 定义 E / T ====
            // E：单位 token 能耗（mJ/token）
            double e_per_token = m.energy / static_cast<double>(TOTAL_TOKENS);

            // T：稳态 per-token latency（s/token，越小越好）
            double steady_lat = 1.0 / m.steady_ts;

            energies_per_token.push_back(e_per_token);
            latencies_per_token.push_back(steady_lat);

            std::cout << "  sample " << i << " : E_total=" << m.energy << " mJ"
                      << ", E_per_token=" << e_per_token << " mJ/token"
                      << ", total_lat=" << m.total_latency_s << " s"
                      << " (FTL=" << m.ftl_s << " s, steady_ts=" << m.steady_ts << " tok/s"
                      << ", steady_lat=" << steady_lat << " s/token)\n";
        }

        double avgE   = mean(energies_per_token);
        double avgLat = mean(latencies_per_token);
        std::cout << "[WINDOW " << w << "] cfg=(" << curFreqKHz << " kHz, n=" << curThreads
                  << "), avg_E_per_token=" << avgE << " mJ/token"
                  << ", avg_steady_lat=" << avgLat << " s/token\n";

        // 本窗口观测到的最大温度（占位）
        double windowMaxTempC = read_window_max_temp_c();

        // 把本窗口的样本喂给优化器：内部会更新 history + 选择下一个配置
        // 带温度版本：若你不想管温度，也可以用 optimizer.postBatch(energies_per_token, latencies_per_token);
        optimizer.postBatch(energies_per_token, latencies_per_token, windowMaxTempC);

        // 下一窗口开始时，直接用 optimizer.currentConfig() 拿新配置即可
    }

    // -------- 6. 收尾 --------
    llama_backend_free();
    return 0;
}
