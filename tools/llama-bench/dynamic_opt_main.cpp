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

#include <fcntl.h>   // open, O_WRONLY, O_CLOEXEC
#include <unistd.h>  // write, close

#include <cerrno>    // errno
#include <chrono>
#include <cstdint>   // uint64_t
#include <cstdio>    // fprintf, stderr
#include <cstring>   // strerror
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// =================== 小工具：时间戳 ===================

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

// =================== 一些 sysfs 读写工具 ===================

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

// 写 long long
static inline bool write_ll(const char * path, long long v) {
    std::ofstream f(path);
    if (!f.good()) {
        return false;
    }
    f << v;
    return !f.fail();
}

// 读取某个 policy 的 scaling_available_frequencies
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

// ============== 实际设置 CPU 频点（可按需替换） ==============
// 这里用的是“锁 min/max 到同一个频率”的方式，你之前已经在用这套逻辑

static bool apply_cpu_freq_khz(int policy, int f) {
    std::cout << "[DVFS] set CPU policy" << policy << " freq to " << f << " kHz\n";
    if (f < 0) {
        return true;  // -1 表示“不改频率”
    }

    std::string base     = "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(policy);
    std::string path_min = base + "/scaling_min_freq";
    std::string path_max = base + "/scaling_max_freq";

    long long cur_min = -1, cur_max = -1;
    (void) read_ll(path_min.c_str(), cur_min);
    (void) read_ll(path_max.c_str(), cur_max);

    bool ok1 = false, ok2 = false;
    if (cur_min < 0 || cur_max < 0) {
        // 读失败了就直接尝试写
        ok1 = write_ll(path_min.c_str(), f);
        ok2 = write_ll(path_max.c_str(), f);
    } else if (f < cur_min) {
        ok1 = write_ll(path_min.c_str(), f);
        ok2 = write_ll(path_max.c_str(), f);
    } else {
        ok2 = write_ll(path_max.c_str(), f);
        ok1 = write_ll(path_min.c_str(), f);
    }

    // 若存在 setspeed，可按需再试一次（通常需要 userspace governor）
    // std::string setspeed = base + "/scaling_setspeed";
    // if (file_exists(setspeed.c_str())) {
    //     (void) write_ll(setspeed.c_str(), f);
    // }

    if (!ok1 || !ok2) {
        std::fprintf(stderr, "warning: set policy%d freq=%d failed (need root?)\n", policy, f);
    }
    return ok1 && ok2;
}

// 读取当前真实 CPU 频率（例如 policy4）
static int read_current_cpu_freq_khz() {
    long long v  = 0;
    bool      ok = read_ll("/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq", v);
    if (!ok || v <= 0) {
        return -1;  // 读取失败，用 -1 作为占位
    }
    return (int) v;
}

// llama 日志回调：静音
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
            needHeader = true;  // 文件不存在
        } else {
            auto size = fin.tellg();
            if (size <= 0) {
                needHeader = true;  // 空文件
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
                    << "avg_energy_mJ,"  // 这里写的是“窗口能量 mJ”，命名沿用之前
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

// ============== 工厂方法：根据 algo_flag 创建优化器 ==============
// algo_flag 支持： "dvfs" / "grid" / "linear" / "neighbor" / "bayes" / "mab"

static std::unique_ptr<CpuFreqOptimizerBase> create_optimizer(const std::string &      algo_flag,
                                                              double                   alpha,
                                                              const std::vector<int> & freqLevelsKHz,
                                                              const std::vector<int> & threadLevels,
                                                              size_t                   samplesPerWindow,
                                                              const char *&            algo_name_out) {
    if (algo_flag == "dvfs") {
        // 系统 DVFS 基线：不创建优化器（nullptr），在 main 里特殊处理
        algo_name_out = "DVFS_system";
        return nullptr;
    }

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

static bool write_wakelock(const char * path, const char * name) {
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        perror("open");
        return false;
    }
    char buf[128];
    int  n = snprintf(buf, sizeof(buf), "%s\n", name);
    if (write(fd, buf, n) != n) {
        perror("write");
        close(fd);
        return false;
    }
    close(fd);
    return true;
}

// ============== 主流程：真实窗口 + 在线优化/基线 ==============

int main(int argc, char ** argv) {
    if (!write_wakelock("/sys/power/wake_lock", "llmbench")) {
        std::fprintf(stderr, "acquire wakelock failed (need root/SELinux permissive)\n");
    }
    llama_log_set(llama_null_log_callback, nullptr);

    // -------- 1. 基础初始化（一次性） --------
    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // 模型路径：argv[1]，算法标志：argv[2]
    std::string model_path = "/data/local/tmp/cpp/Qwen3-0.6B-Q4_0.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::string algo_flag = "grid";  // 默认 grid
    if (argc > 2) {
        algo_flag = argv[2];         // "dvfs" / "grid" / "linear" / "neighbor" / "bayes" / "mab"
    }
    std::cout << "[MAIN] model_path = " << model_path << ", algo_flag = " << algo_flag << std::endl;

    // -------- 2. 创建 LlamaRunner --------
    const int n_ctx        = 512;  // 至少 >= 64 + 32
    const int n_batch      = 2048;
    const int init_threads = 4;    // 初始线程数，后面会被动态修改

    LlamaRunner runner(model_path, n_ctx, n_batch, init_threads);

    // -------- 3. 定义可选的 CPU 频率档位（policy4） --------
    std::vector<int> freqLevelsKHz = read_available_freqs(4);
    if (freqLevelsKHz.size() >= 8) {
        // 保留最高的几个频点，防止空间过大
        freqLevelsKHz.erase(freqLevelsKHz.begin(), freqLevelsKHz.end() - 8);
    }
    if (freqLevelsKHz.empty()) {
        std::cerr << "[WARN] scaling_available_frequencies empty for policy4, "
                  << "fallback to hard-coded freq levels\n";
        freqLevelsKHz = { 844800, 1190400, 1497600, 1785600, 2073600, 2352000 };
    }

    // -------- 3.1 定义可选的线程数档位 --------
    std::vector<int> threadLevels       = { 1, 2, 3 };
    const int        MAX_WINDOWS        = 50;  // 真实窗口总数
    // 每一“窗口”用多少次推理样本进行统计
    const size_t     SAMPLES_PER_WINDOW = 5;

    // 代价函数权重 alpha：J = alpha * E_norm + (1-alpha) * T_norm
    const double alpha = 0.5;

    // -------- 4. 创建优化器（通过工厂方法） --------
    const char * algo_name = nullptr;
    auto optimizer = create_optimizer(algo_flag, alpha, freqLevelsKHz, threadLevels, SAMPLES_PER_WINDOW, algo_name);

    std::cout << "[MAIN] algo_name = " << (algo_name ? algo_name : "UNKNOWN") << std::endl;

    // -------- 5. 创建功耗采样器 --------
    PowerSampler sampler(
        /*period_ms=*/50,
        /*base=*/"/sys/class/power_supply/battery",
        /*tz_path=*/"/sys/class/thermal/thermal_zone37/temp",
        /*log_path=*/""  // 不单独落盘 trace，窗口内 snapshot 即可
    );

    // -------- 6. 打开 CSV --------
    ensure_window_csv_opened("/data/local/tmp/cpp/window_metrics.csv");

    // -------- 7. 主循环：真实请求 + 每窗口调一次优化器/记录 DVFS --------

    sampler.start();

    for (int w = 0; w < MAX_WINDOWS; ++w) {
        // ★ 记录窗口开始时的系统时间
        auto wall_now = std::chrono::system_clock::now();
        auto wall_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(wall_now.time_since_epoch()).count();

        // 格式化成本地可读时间 "YYYY-MM-DD HH:MM:SS"
        std::time_t tt      = std::chrono::system_clock::to_time_t(wall_now);
        char        buf[32] = { 0 };
        std::tm     tm_local;
        // Android / bionic 支持 localtime_r
        localtime_r(&tt, &tm_local);
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_local);
        std::string wall_str(buf);

        int curFreqKHz = -1;
        int curThreads = 0;
        int freqIdx    = -1;
        int threadIdx  = -1;

        if (optimizer) {
            // 动态优化算法：由优化器给出当前配置
            CpuFreqConfig cfg = optimizer->currentConfig();
            freqIdx           = cfg.freqIdx;
            threadIdx         = cfg.threadIdx;
            curFreqKHz        = freqLevelsKHz[freqIdx];
            curThreads        = threadLevels[threadIdx];
        } else {
            // DVFS 基线：不主动控频，只设定一个固定线程数（比如 4）
            freqIdx    = -1;
            threadIdx  = -1;
            curFreqKHz = -1;  // target_freq 无意义
            int size   = sizeof(threadLevels) / sizeof(threadLevels[0]);
            curThreads = threadLevels[size - 1];
        }

        std::cout << "\n==================== WINDOW " << w;
        if (optimizer) {
            std::cout << " : freqIdx=" << freqIdx << " (" << curFreqKHz << " kHz)"
                      << ", threadIdx=" << threadIdx << " (n=" << curThreads << ")";
        } else {
            std::cout << " : DVFS baseline, n_threads=" << curThreads;
        }
        std::cout << " ====================\n";

        // 下发 DVFS 配置 + 线程配置
        if (optimizer) {
            // 只有在线优化算法才写 sysfs 控频
            apply_cpu_freq_khz(4, curFreqKHz);
            runner.set_freq_ghz(curFreqKHz / 1e6);  // kHz -> GHz
        } else {
            // DVFS：完全交给系统 governor，不写任何频率节点
            int realFreq = read_current_cpu_freq_khz();
            runner.set_freq_ghz(realFreq > 0 ? realFreq / 1e6 : 0.0);
        }
        runner.set_num_threads(curThreads);

        // 在当前配置下，跑 SAMPLES_PER_WINDOW 次 64+32 推理
        std::vector<double> steady_lats;  // 稳态 s/token
        std::vector<double> total_lats;   // 总延迟
        std::vector<double> ftls;         // first token latency
        std::vector<double> overall_tps;  // overall tok/s
        std::vector<double> steady_tps;   // steady tok/s

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

            steady_lats.push_back(steady_lat);
            total_lats.push_back(m.total_latency_s);
            ftls.push_back(m.ftl_s);
            overall_tps.push_back(m.overall_ts);
            steady_tps.push_back(m.steady_ts);

            std::cout << "  sample " << i << " : total_lat=" << m.total_latency_s << " s"
                      << " (FTL=" << m.ftl_s << " s, steady_ts=" << m.steady_ts << " tok/s"
                      << ", steady_lat=" << steady_lat << " s/token)\n";
        }

        uint64_t t_win_end   = lr_now_ns();
        auto     snap        = sampler.snapshot();
        double   window_mJ   = PowerSampler::integrate_mJ(snap, t_win_start, t_win_end);
        double   max_temp_dC = PowerSampler::max_temp_dC(snap, t_win_start, t_win_end);
        double   maxTempC    = max_temp_dC / 10.0;  // deci-℃ -> ℃

        double avgSteadyLat  = mean(steady_lats);
        double avgTotalLat   = mean(total_lats);
        double avgFtl        = mean(ftls);
        double avgOverallTps = mean(overall_tps);
        double avgSteadyTps  = mean(steady_tps);

        std::cout << "[WINDOW " << w << "] cfg=(" << curFreqKHz << " kHz, n=" << curThreads << "), "
                  << "windowEnergy=" << window_mJ << " mJ"
                  << ", avg_steady_lat=" << avgSteadyLat << " s/token"
                  << ", avg_total_lat=" << avgTotalLat << " s"
                  << ", avg_FTL=" << avgFtl << " s"
                  << ", avg_overall_ts=" << avgOverallTps << " tok/s"
                  << ", avg_steady_ts=" << avgSteadyTps << " tok/s"
                  << ", maxTemp=" << maxTempC << " C\n";

        // 读取真实频率（可能与我们下发的 target 不一致）
        int realFreqKHz = read_current_cpu_freq_khz();

        // 记录优化器开销
        double optimizerMs    = 0.0;
        size_t numSamplesUsed = steady_lats.size();

        if (optimizer && numSamplesUsed > 0) {
            auto tOptStart = std::chrono::steady_clock::now();
            optimizer->postBatch(window_mJ, steady_lats, maxTempC);
            auto tOptEnd = std::chrono::steady_clock::now();
            optimizerMs  = std::chrono::duration_cast<std::chrono::microseconds>(tOptEnd - tOptStart).count() / 1000.0;

            // 修正后的输出
            std::cout << "[WINDOW " << w << "] algo_name=" << algo_name << " optimizer_time_ms=" << optimizerMs
                      << " samples=" << numSamplesUsed << " realFreqKHz=" << realFreqKHz << std::endl;
        } else {
            std::cout << "[WINDOW " << w << "] DVFS baseline (no optimizer)"
                      << " samples=" << numSamplesUsed << " realFreqKHz=" << realFreqKHz << std::endl;
        }

        // 写入 CSV（window_metrics.csv）
        if (g_windowCsv.is_open()) {
            g_windowCsv << w << ","                                    // window_id
                        << wall_str << ","                             // ★ 可读时间
                        << wall_ms << ","                              // ★ Unix ms
                        << (algo_name ? algo_name : "UNKNOWN") << ","  // algo_name
                        << alpha << ","                                // alpha
                        << freqIdx << ","                              // freq_idx（DVFS 下为 -1）
                        << threadIdx << ","                            // thread_idx（DVFS 下为 -1）
                        << curFreqKHz << ","                           // target_freq_khz（DVFS 下为 -1）
                        << realFreqKHz << ","                          // real_freq_khz
                        << curThreads << ","                           // n_threads
                        << numSamplesUsed << ","                       // samples
                        << window_mJ << ","                            // avg_energy_mJ（这里实为 windowEnergy）
                        << avgSteadyLat << ","                         // avg_steady_lat_s_per_token
                        << avgTotalLat << ","                          // avg_total_lat_s
                        << avgFtl << ","                               // avg_ftl_s
                        << avgOverallTps << ","                        // avg_overall_ts
                        << avgSteadyTps << ","                         // avg_steady_ts
                        << optimizerMs                                 // optimizer_time_ms
                        << "\n";
            g_windowCsv.flush();
        }

        // 下一窗口开始时，动态算法直接用 optimizer->currentConfig() 拿新配置；
        // DVFS 模式则继续走系统自带 governor。
    }

    sampler.stop();

    // -------- 8. 收尾 --------
    if (g_windowCsv.is_open()) {
        g_windowCsv.close();
    }

    llama_backend_free();
    if (!write_wakelock("/sys/power/wake_unlock", "llmbench")) {
        fprintf(stderr, "return wakelock failed (need root/SELinux permissive)\n");
    }

    return 0;
}
