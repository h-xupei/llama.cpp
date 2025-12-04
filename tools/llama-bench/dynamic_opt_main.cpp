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
#include "power_sample.cpp"

#include <chrono>
#include <cstdio>  // fprintf, stderr
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static inline uint64_t lr_now_ns() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
}

// ============== 简单 helper：算平均值 ==============
static double mean(const std::vector<double> & v) {
    if (v.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / static_cast<double>(v.size());
}

// 读整个文件为字符串
static inline bool read_str(const char * path, std::string & out) {
    std::ifstream f(path);
    if (!f.good()) {
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// 读 long long
static inline bool read_ll(const char * path, long long & out) {
    std::ifstream f(path);
    if (!f.good()) {
        return false;
    }
    long long v;
    f >> v;
    if (!f.fail()) {
        out = v;
        return true;
    }
    return false;
}

static inline bool write_ll(const char * path, long long v) {
    std::ofstream f(path);
    if (!f.good()) {
        return false;
    }
    f << v;
    return !f.fail();
}

static inline std::vector<int> read_available_freqs(int policy) {
    std::vector<int> v;
    std::string      s;
    std::string      path =
        "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(policy) + "/scaling_available_frequencies";
    if (!read_str(path.c_str(), s)) {
        return v;
    }
    std::stringstream ss(s);
    long long         x;
    while (ss >> x) {
        v.push_back((int) x);
    }
    return v;
}

// ============== 占位：实际设置 CPU 频点 ==============
static bool apply_cpu_freq_khz(int policy, int f) {
    // TODO: 用你真机上的 DVFS 控制逻辑替换
    std::cout << "[DVFS] set CPU freq to " << f << " kHz" << std::endl;
    if (f < 0) {
        return true;  // -1 表示不改
    }
    std::string base     = "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(policy);
    std::string path_min = base + "/scaling_min_freq";
    std::string path_max = base + "/scaling_max_freq";

    long long cur_min = -1, cur_max = -1;
    (void) read_ll(path_min.c_str(), cur_min);
    (void) read_ll(path_max.c_str(), cur_max);

    bool ok1 = false, ok2 = false;
    if (f < cur_min) {
        ok1 = write_ll(path_min.c_str(), f);
        ok2 = write_ll(path_max.c_str(), f);
    } else {
        ok2 = write_ll(path_max.c_str(), f);
        ok1 = write_ll(path_min.c_str(), f);
    }

    // 若存在 setspeed，尝试一下（通常需 userspace governor，但不强制切换）
    // if (file_exists((base + "/scaling_setspeed").c_str())) {
    //     (void) write_ll((base + "/scaling_setspeed").c_str(), f);
    // }

    if (!ok1 || !ok2) {
        std::fprintf(stderr, "warning: set policy%d freq=%d failed (need root?)\n", policy, f);
    }
    return ok1 && ok2;
}

// 读取当前真实 CPU 频率（示例：policy4）
static int read_current_cpu_freq_khz() {
    long long v  = 0;
    bool      ok = read_ll("/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq", v);
    if (!ok) {
        // 读取失败时给个占位值，便于后期分析看出来
        return -1;
    }
    return (int) v;
}

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

// ============== CSV 全局句柄 ==============

static std::ofstream g_windowCsv;
static bool          g_windowCsvInited = false;

// 只在文件新建 / 为空时写表头，多次运行追加时不会重复表头
static void ensure_window_csv_opened(const std::string & path) {
    if (g_windowCsvInited) {
        return;
    }

    bool needHeader = false;
    {
        std::ifstream fin(path, std::ios::in | std::ios::ate);
        if (!fin.is_open()) {
            // 文件不存在 ⇒ 需要写表头
            needHeader = true;
        } else {
            auto size = fin.tellg();
            if (size <= 0) {
                // 空文件 ⇒ 需要写表头
                needHeader = true;
            }
        }
    }

    // 以 append 方式打开，不截断历史数据
    g_windowCsv.open(path, std::ios::out | std::ios::app);
    if (!g_windowCsv.is_open()) {
        std::cerr << "[ERROR] cannot open " << path << " for write\n";
        return;
    }

    if (needHeader) {
        g_windowCsv << "window_id,"
                    << "algo_name,"
                    << "alpha,"
                    << "freq_idx,"
                    << "thread_idx,"
                    << "target_freq_khz,"
                    << "real_freq_khz,"
                    << "n_threads,"
                    << "samples,"
                    << "avg_energy_mJ,"
                    << "avg_steady_lat_s_per_token,"
                    << "avg_total_lat_s,"
                    << "avg_ftl_s,"
                    << "avg_overall_ts,"
                    << "avg_steady_ts,"
                    << "optimizer_time_ms"
                    << "\n";
    }

    g_windowCsvInited = true;
}

// ============== 工厂方法：根据 algo_flag 创建优化器，并设置 algo_name ==============

static std::unique_ptr<CpuFreqOptimizerBase> create_optimizer(const std::string &      algo_flag,
                                                              double                   alpha,
                                                              const std::vector<int> & freqLevelsKHz,
                                                              const std::vector<int> & threadLevels,
                                                              size_t                   samplesPerWindow,
                                                              const char *&            algo_name_out) {
    // 注意：这里不做复杂的参数解析，只按字符串精确匹配
    if (algo_flag == "grid") {
        algo_name_out = "GridSearch";
        return std::make_unique<GridSearchCpuFreqOptimizer>(alpha, freqLevelsKHz, threadLevels, samplesPerWindow);
    }

    if (algo_flag == "linear") {
        algo_name_out = "LinearSearch";
        return std::make_unique<LinearSearchOptimizer>(alpha, freqLevelsKHz, threadLevels, samplesPerWindow);
    }

    if (algo_flag == "neighbor") {
        algo_name_out = "NeighborSearch";
        return std::make_unique<NeighborSearchOptimizer>(alpha, freqLevelsKHz, threadLevels, samplesPerWindow);
    }

    if (algo_flag == "bayes" || algo_flag == "bayesian") {
        algo_name_out = "Bayesian";
        return std::make_unique<BayesianCpuFreqOptimizer>(alpha, freqLevelsKHz, threadLevels, samplesPerWindow);
    }

    // 默认：MAB 多维优化
    algo_name_out = "MABMultiDim";
    return std::make_unique<MABMultiDimCpuFreqOptimizer>(alpha, freqLevelsKHz, threadLevels, samplesPerWindow);
}

// ============== 主流程：只有“真实窗口”，每窗口结束调一次优化器 ==============

int main(int argc, char ** argv) {
    llama_log_set(llama_null_log_callback, nullptr);

    // -------- 1. 基础初始化（一次性） --------
    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // 模型路径，可以从命令行传，也可以写死
    std::string model_path = "/data/local/tmp/cpp/Qwen3-0.6B-Q4_0.gguf";
    if (argc > 1) {
        // 保持之前行为：argv[1] 是模型路径，其它参数忽略
        model_path = argv[1];
    }

    // -------- 2. 创建 LlamaRunner --------
    const int n_ctx        = 512;  // 至少 >= 64 + 32
    const int n_batch      = 2048;
    const int init_threads = 4;    // 初始线程数，后面会被动态修改

    LlamaRunner runner(model_path, n_ctx, n_batch, init_threads);

    // -------- 3. 定义可选的 CPU 频率档位 --------
    std::vector<int> freqLevelsKHz = read_available_freqs(4);
    if (freqLevelsKHz.size() >= 8) {
        freqLevelsKHz.erase(freqLevelsKHz.begin(), freqLevelsKHz.end() - 8);
        // 现在 freqLevelsKHz 只包含后8个元素
    }
    if (freqLevelsKHz.empty()) {
        std::cerr << "[WARN] scaling_available_frequencies empty for policy4, "
                  << "fallback to hard-coded freq levels\n";
        freqLevelsKHz = { 844800, 1190400, 1497600, 1785600, 2073600, 2352000 };
    }

    // -------- 3.1 定义可选的线程数档位 --------
    std::vector<int> threadLevels = { 1, 2, 3 };

    // 每一“窗口”用多少次推理样本进行统计
    const size_t SAMPLES_PER_WINDOW = 10;

    // 代价函数权重 alpha：J = alpha * E_norm + (1-alpha) * T_norm
    const double alpha = 0.5;

    // -------- 4. 创建优化器（通过工厂方法） --------
    // 这里简单用一个字符串切换算法，后续要换算法只改 algo_flag 即可：
    // 可选："grid" / "linear" / "neighbor" / "bayes" / "mab"
    std::string  algo_flag = "grid";  // 当前用 GridSearch，如果要改 MAB，就写 "mab"
    const char * algo_name = nullptr;
    auto optimizer = create_optimizer(algo_flag, alpha, freqLevelsKHz, threadLevels, SAMPLES_PER_WINDOW, algo_name);
    PowerSampler sampler(
        /*period_ms=*/50,
        /*base=*/"/sys/class/power_supply/battery",
        /*tz_path=*/"/sys/class/thermal/thermal_zone37/temp",
        /*log_path=*/""  // 或者 ""
    );
    // 打开 CSV（用绝对路径，避免当前工作目录不是 /data/local/tmp/cpp）
    ensure_window_csv_opened("/data/local/tmp/cpp/window_metrics.csv");

    // -------- 5. 主循环：真实请求 + 每窗口调一次优化器 --------
    const int MAX_WINDOWS = 100;  // 真实窗口总数，随便写个上限
    sampler.start();
    for (int w = 0; w < MAX_WINDOWS; ++w) {
        // 当前窗口配置：由优化器给出
        CpuFreqConfig cfg = optimizer->currentConfig();

        int curFreqKHz = freqLevelsKHz[cfg.freqIdx];
        int curThreads = threadLevels[cfg.threadIdx];

        std::cout << "\n==================== WINDOW " << w << " : freqIdx=" << cfg.freqIdx << " (" << curFreqKHz
                  << " kHz)"
                  << ", threadIdx=" << cfg.threadIdx << " (n=" << curThreads << ")"
                  << " ====================\n";

        // 下发 DVFS 配置 + 线程配置
        apply_cpu_freq_khz(4, curFreqKHz);
        runner.set_freq_ghz(curFreqKHz / 1e6);  // kHz -> GHz
        runner.set_num_threads(curThreads);

        // 在当前配置下，跑 SAMPLES_PER_WINDOW 次 64+32 推理
        std::vector<double> energies;
        std::vector<double> steady_lats;  // 稳态 s/token
        std::vector<double> total_lats;   // 总延迟
        std::vector<double> ftls;         // first token latency
        std::vector<double> overall_tps;  // overall tok/s
        std::vector<double> steady_tps;   // steady tok/s

        energies.reserve(SAMPLES_PER_WINDOW);
        steady_lats.reserve(SAMPLES_PER_WINDOW);
        total_lats.reserve(SAMPLES_PER_WINDOW);
        ftls.reserve(SAMPLES_PER_WINDOW);
        overall_tps.reserve(SAMPLES_PER_WINDOW);
        steady_tps.reserve(SAMPLES_PER_WINDOW);

        uint64_t t_win_start = lr_now_ns();
        for (size_t i = 0; i < SAMPLES_PER_WINDOW; ++i) {
            auto m = runner.run_one_request(/*n_prompt=*/64, /*n_gen=*/32);

            if (m.steady_ts <= 0.0) {
                std::cerr << "[WIN " << w << "] warning: steady_ts <= 0, skip sample " << i << " cfg=(" << curFreqKHz
                          << " kHz, n=" << curThreads << ")\n";
                continue;
            }

            double steady_lat = 1.0 / m.steady_ts;  // s/token，越小越好

                                                    //            energies.push_back(m.energy);
            steady_lats.push_back(steady_lat);
            total_lats.push_back(m.total_latency_s);
            ftls.push_back(m.ftl_s);
            overall_tps.push_back(m.overall_ts);
            steady_tps.push_back(m.steady_ts);

            std::cout << "  sample "
                      << i
                      //                      << " : E=" << m.energy << " mJ"
                      << ", total_lat=" << m.total_latency_s << " s"
                      << " (FTL=" << m.ftl_s << " s, steady_ts=" << m.steady_ts << " tok/s"
                      << ", steady_lat=" << steady_lat << " s/token)\n";
        }
        uint64_t t_win_end     = lr_now_ns();
        auto     snap          = sampler.snapshot();
        double   win_mJ        = PowerSampler::integrate_mJ(snap, t_win_start, t_win_end);
        double   max_temp_dC   = PowerSampler::max_temp_dC(snap, t_win_start, t_win_end);
        double   maxTempC      = max_temp_dC / 10.0;  // deci-℃ -> ℃
        double   avgE          = win_mJ;              // 每次请求平均能量（mJ / request）
                                                      //        double avgE          = mean(energies);
        double   avgSteadyLat  = mean(steady_lats);
        double   avgTotalLat   = mean(total_lats);
        double   avgFtl        = mean(ftls);
        double   avgOverallTps = mean(overall_tps);
        double   avgSteadyTps  = mean(steady_tps);

        std::cout << "[WINDOW " << w << "] cfg=(" << curFreqKHz << " kHz, n=" << curThreads << "), "
                  << "avgE=" << avgE << " mJ"
                  << ", avg_steady_lat=" << avgSteadyLat << " s/token"
                  << ", avg_total_lat=" << avgTotalLat << " s"
                  << ", avg_FTL=" << avgFtl << " s"
                  << ", avg_overall_ts=" << avgOverallTps << " tok/s"
                  << ", avg_steady_ts=" << avgSteadyTps << " tok/s\n";

        // 读取真实频率（可能与我们下发的 target 不一致）
        int realFreqKHz = read_current_cpu_freq_khz();

        // 记录优化器开销
        auto tOptStart = std::chrono::steady_clock::now();
        optimizer->postBatch(avgE, steady_lats, maxTempC);
        auto   tOptEnd = std::chrono::steady_clock::now();
        double optimizerMs =
            std::chrono::duration_cast<std::chrono::microseconds>(tOptEnd - tOptStart).count() / 1000.0;
        size_t numSamplesUsed = steady_lats.size();

        std::cout << "[WINDOW " << w << "] optimizer_time_ms=" << optimizerMs << " (samples=" << numSamplesUsed
                  << ")\n";

        // 写入 CSV（window_metrics.csv）
        if (g_windowCsv.is_open()) {
            g_windowCsv << w << ","                                    // window_id
                        << (algo_name ? algo_name : "UNKNOWN") << ","  // algo_name
                        << alpha << ","                                // alpha
                        << cfg.freqIdx << ","                          // freq_idx
                        << cfg.threadIdx << ","                        // thread_idx
                        << curFreqKHz << ","                           // target_freq_khz
                        << realFreqKHz << ","                          // real_freq_khz
                        << curThreads << ","                           // n_threads
                        << numSamplesUsed << ","                       // samples
                        << avgE << ","                                 // avg_energy_mJ
                        << avgSteadyLat << ","                         // avg_steady_lat_s_per_token
                        << avgTotalLat << ","                          // avg_total_lat_s
                        << avgFtl << ","                               // avg_ftl_s
                        << avgOverallTps << ","                        // avg_overall_ts
                        << avgSteadyTps << ","                         // avg_steady_ts
                        << optimizerMs                                 // optimizer_time_ms
                        << "\n";
            g_windowCsv.flush();
        }

        // 下一窗口开始时，直接用 optimizer->currentConfig() 拿新配置即可
    }
    sampler.stop();
    // -------- 6. 收尾 --------
    if (g_windowCsv.is_open()) {
        g_windowCsv.close();
    }

    llama_backend_free();
    return 0;
}
