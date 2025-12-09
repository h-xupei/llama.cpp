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
#include <ctime>     // time_t, localtime_r, strftime
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
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

// 尝试让 sysfs 节点变成可写：
// 1）本来就可写 -> 直接返回 true
// 2）不可写 -> chmod 0664，再检查一次
static bool ensure_sysfs_writable(const std::string & path) {
    // 已经可写就不用管
    if (access(path.c_str(), W_OK) == 0) {
        return true;
    }

    // 直接粗暴一点，改成 0664
    if (chmod(path.c_str(), 0664) != 0) {
        std::fprintf(stderr, "[DVFS] chmod(%s, 0664) failed: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }

    // 再确认一次
    if (access(path.c_str(), W_OK) != 0) {
        std::fprintf(stderr, "[DVFS] %s still not writable after chmod: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }

    return true;
}

// ============== 实际设置 CPU 频点 ==============

static bool apply_cpu_freq_khz(int policy, int f) {
    std::cout << "[DVFS] set CPU policy" << policy << " freq to " << f << " kHz\n";
    if (f < 0) {
        return true;  // -1 表示"不改频率"
    }

    std::string policy_path = "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(policy);
    std::string path_min    = policy_path + "/scaling_min_freq";
    std::string path_max    = policy_path + "/scaling_max_freq";

    // 直接用一条 su 命令：先改权限，再设置频率
    std::string cmd =
        "su -c '"
        "chmod 666 " +
        path_min + " " + path_max +
        " && "
        "echo " +
        std::to_string(f) + " > " + path_max +
        " && "
        "echo " +
        std::to_string(f) + " > " + path_min + "' 2>/dev/null";

    int ret = system(cmd.c_str());

    if (ret != 0) {
        std::fprintf(stderr, "warning: set policy%d freq=%d failed\n", policy, f);
        return false;
    }

    return true;
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

    g_windowCsv.open(path, std::ios::out | std::ios::app);
    if (!g_windowCsv.is_open()) {
        std::cerr << "[ERROR] cannot open " << path << " for write\n";
        return;
    }

    if (needHeader) {
        g_windowCsv << "window_id,"
                    << "wall_time_str,"
                    << "wall_time_ms,"
                    << "algo_name,"
                    << "alpha,"
                    << "freq_idx,"
                    << "thread_idx,"
                    << "target_freq_khz,"
                    << "real_freq_khz,"
                    << "n_threads,"
                    << "samples,"
                    << "window_start_temp_c,"
                    << "window_max_temp_c,"
                    << "avg_energy_mJ,"
                    << "avg_fixed_mJ,"
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

static std::unique_ptr<CpuFreqOptimizerBase> create_optimizer(const std::string &      algo_flag,
                                                              double                   alpha,
                                                              const std::vector<int> & freqLevelsKHz,
                                                              const std::vector<int> & threadLevels,
                                                              size_t                   samplesPerWindow,
                                                              const char *&            algo_name_out) {
    if (algo_flag == "dvfs") {
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

// ============== helper：温度冷却到 targetC 以下 ==============
// 使用 power_sample.cpp 里的 ps_file_exists / ps_read_temp_to_dC

static bool wait_cool_down(const char * tz_path, double targetC, int timeout_sec) {
    if (!ps_file_exists(tz_path)) {
        std::fprintf(stderr, "[COOL] thermal path %s not found, skip cooldown\n", tz_path);
        return false;
    }

    const int interval_ms = 1000;
    int       waited_ms   = 0;

    std::printf("[COOL] start cooldown: target <= %.1f C, timeout=%d s\n", targetC, timeout_sec);

    while (waited_ms < timeout_sec * 1000) {
        long long dC = 0;
        if (!ps_read_temp_to_dC(tz_path, dC)) {
            std::fprintf(stderr, "[COOL] read temp failed from %s, break\n", tz_path);
            break;
        }
        double tempC = dC / 10.0;
        std::printf("[COOL] current temp = %.1f C\n", tempC);

        if (tempC <= targetC) {
            std::printf("[COOL] cooled down to %.1f C (<= %.1f C), continue\n", tempC, targetC);
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        waited_ms += interval_ms;
    }

    std::printf("[COOL] cooldown timeout or failed, continue anyway\n");
    return false;
}

// ============== 静态场景：跑一个优化器一轮 ==============

static void run_static_for_algo(const std::string &      algo_flag,
                                LlamaRunner &            runner,
                                PowerSampler &           sampler,
                                const std::vector<int> & freqLevelsKHz,
                                const std::vector<int> & threadLevels,
                                int                      MAX_WINDOWS,
                                size_t                   SAMPLES_PER_WINDOW,
                                double                   alpha) {
    const char * algo_name = nullptr;
    auto optimizer = create_optimizer(algo_flag, alpha, freqLevelsKHz, threadLevels, SAMPLES_PER_WINDOW, algo_name);

    std::cout << "\n\n========== STATIC SCENARIO RUN, ALGO = " << (algo_name ? algo_name : algo_flag.c_str())
              << " ==========\n";

    // 注意：不启用温度惩罚/FTL 约束，纯静态基线
    if (optimizer) {
        optimizer->set_thermal_enabled(false);
    }
    for (int w = 0; w < MAX_WINDOWS; ++w) {
        int curFreqKHz = -1;
        int curThreads = 0;
        int freqIdx    = -1;
        int threadIdx  = -1;

        if (optimizer) {
            CpuFreqConfig cfg = optimizer->currentConfig();
            freqIdx           = cfg.freqIdx;
            threadIdx         = cfg.threadIdx;
            curFreqKHz        = freqLevelsKHz[freqIdx];
            curThreads        = threadLevels[threadIdx];
        } else {
            // DVFS baseline：不控频，线程数用最大
            freqIdx    = -1;
            threadIdx  = -1;
            curFreqKHz = -1;
            curThreads = threadLevels.back();
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
            apply_cpu_freq_khz(4, curFreqKHz);
        }
        runner.set_num_threads(curThreads);
        // 每个优化器预热一次请求
        runner.run_one_request(/*n_prompt=*/64, /*n_gen=*/32);
        // 记录窗口开始时的系统时间
        auto wall_now = std::chrono::system_clock::now();
        auto wall_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(wall_now.time_since_epoch()).count();

        std::time_t tt      = std::chrono::system_clock::to_time_t(wall_now);
        char        buf[32] = { 0 };
        std::tm     tm_local;
        localtime_r(&tt, &tm_local);
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_local);
        std::string wall_str(buf);

        std::vector<double> steady_lats;
        std::vector<double> total_lats;
        std::vector<double> ftls;
        std::vector<double> overall_tps;
        std::vector<double> steady_tps;

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

            double steady_lat = 1.0 / m.steady_ts;  // s/token

            steady_lats.push_back(steady_lat);
            total_lats.push_back(m.total_latency_s);
            ftls.push_back(m.ftl_s);
            overall_tps.push_back(m.overall_ts);
            steady_tps.push_back(m.steady_ts);

            std::cout << "  sample " << i << " : total_lat=" << m.total_latency_s << " s"
                      << " (FTL=" << m.ftl_s << " s, steady_ts=" << m.steady_ts << " tok/s"
                      << ", steady_lat=" << steady_lat << " s/token)\n";
        }
        int realFreqKHz = read_current_cpu_freq_khz();
        runner.set_freq_ghz(realFreqKHz > 0 ? realFreqKHz / 1e6 : 0.0);

        uint64_t t_win_end   = lr_now_ns();
        auto     snap        = sampler.snapshot();
        double   end_mJ      = PowerSampler::integrate_mJ(snap, t_win_start, t_win_end);
        double   max_temp_dC = PowerSampler::max_temp_dC(snap, t_win_start, t_win_end);
        double   maxTempC    = max_temp_dC / 10.0;
        double   startTempC  = PowerSampler::first_temp_dC_after(snap, t_win_start) / 10.0;
        double   basePower   = PowerSampler::first_power_after(snap, t_win_start);
        double   dt_s        = (double) (t_win_end - t_win_start) / 1e9;
        double   start_mJ    = basePower * dt_s;
        double   window_mJ   = end_mJ;
        double   fixed_mJ    = PowerSampler::fake_energy_mJ(realFreqKHz, curThreads, dt_s);

        double avgSteadyLat  = mean(steady_lats);
        double avgTotalLat   = mean(total_lats);
        double avgFtl        = mean(ftls);
        double avgOverallTps = mean(overall_tps);
        double avgSteadyTps  = mean(steady_tps);

        std::cout << "[WINDOW " << w << "] cfg=(" << curFreqKHz << " kHz, n=" << curThreads << "), "
                  << "windowEnergy=" << window_mJ << " mJ"
                  << ", fixedEnergy=" << fixed_mJ << " mJ"
                  << ", avg_steady_lat=" << avgSteadyLat << " s/token"
                  << ", avg_total_lat=" << avgTotalLat << " s"
                  << ", avg_FTL=" << avgFtl << " s"
                  << ", avg_overall_ts=" << avgOverallTps << " tok/s"
                  << ", avg_steady_ts=" << avgSteadyTps << " tok/s"
                  << ", startTemp=" << startTempC << " C"
                  << ", maxTemp=" << maxTempC << " C\n";

        double optimizerMs    = 0.0;
        size_t numSamplesUsed = steady_lats.size();

        if (optimizer && numSamplesUsed > 0) {
            auto tOptStart = std::chrono::steady_clock::now();
            optimizer->postBatch(window_mJ, steady_lats, maxTempC);
            auto tOptEnd = std::chrono::steady_clock::now();
            optimizerMs  = std::chrono::duration_cast<std::chrono::microseconds>(tOptEnd - tOptStart).count() / 1000.0;

            std::cout << "[WINDOW " << w << "] algo_name=" << (algo_name ? algo_name : "UNKNOWN")
                      << " optimizer_time_ms=" << optimizerMs << " samples=" << numSamplesUsed
                      << " realFreqKHz=" << realFreqKHz << std::endl;
        } else {
            std::cout << "[WINDOW " << w << "] DVFS baseline (no optimizer)"
                      << " samples=" << numSamplesUsed << " realFreqKHz=" << realFreqKHz << std::endl;
        }

        if (g_windowCsv.is_open()) {
            g_windowCsv << w << ","                                    // window_id (每个算法内 0..MAX_WINDOWS-1)
                        << wall_str << ","                             // wall_time_str
                        << wall_ms << ","                              // wall_time_ms
                        << (algo_name ? algo_name : "UNKNOWN") << ","  // algo_name
                        << alpha << ","                                // alpha
                        << freqIdx << ","                              // freq_idx
                        << threadIdx << ","                            // thread_idx
                        << curFreqKHz << ","                           // target_freq_khz
                        << realFreqKHz << ","                          // real_freq_khz
                        << curThreads << ","                           // n_threads
                        << numSamplesUsed << ","                       // samples
                        << startTempC << ","                           // window_start_temp_c
                        << maxTempC << ","                             // window_max_temp_c
                        << window_mJ << ","                            // avg_energy_mJ (windowEnergy)
                        << fixed_mJ << ","                             // fixed_energy_mJ (windowEnergy)
                        << avgSteadyLat << ","                         // avg_steady_lat_s_per_token
                        << avgTotalLat << ","                          // avg_total_lat_s
                        << avgFtl << ","                               // avg_ftl_s
                        << avgOverallTps << ","                        // avg_overall_ts
                        << avgSteadyTps << ","                         // avg_steady_ts
                        << optimizerMs                                 // optimizer_time_ms
                        << "\n";
            g_windowCsv.flush();
        }
    }
}

// ============== 主流程：静态场景执行器 ==============

int main(int argc, char ** argv) {
    if (!write_wakelock("/sys/power/wake_lock", "llmbench")) {
        std::fprintf(stderr, "acquire wakelock failed (need root/SELinux permissive)\n");
    }

    const char * tz_path = "/sys/class/thermal/thermal_zone37/temp";

    // 先做一次冷却：保证第一轮实验从较低温度开始
    wait_cool_down(tz_path, /*targetC=*/50.0, /*timeout_sec=*/300);

    llama_log_set(llama_null_log_callback, nullptr);

    // -------- 1. 基础初始化（一次性） --------
    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // 模型路径：argv[1]
    std::string model_path = "/data/local/tmp/cpp/Qwen3-0.6B-Q4_0.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    // algo_flag：argv[2]；默认 "all" 表示依次跑 grid/linear/neighbor/mab/bayes
    std::string algo_flag = "all";
    if (argc > 2) {
        algo_flag = argv[2];
    }

    std::cout << "[MAIN] model_path = " << model_path << ", algo_flag = " << algo_flag << std::endl;

    // -------- 2. 创建 LlamaRunner --------
    const int n_ctx        = 512;
    const int n_batch      = 2048;
    const int init_threads = 4;

    LlamaRunner runner(model_path, n_ctx, n_batch, init_threads);

    // -------- 3. 定义频率/线程档位 --------
    std::vector<int> freqLevelsKHz = read_available_freqs(4);
    if (freqLevelsKHz.size() >= 8) {
        freqLevelsKHz.erase(freqLevelsKHz.begin(), freqLevelsKHz.end() - 8);
    }
    if (freqLevelsKHz.empty()) {
        std::cerr << "[WARN] scaling_available_frequencies empty for policy4, fallback\n";
        freqLevelsKHz = { 844800, 1190400, 1497600, 1785600, 2073600, 2352000 };
    }

    std::vector<int> threadLevels       = { 1, 2, 3 };
    const int        MAX_WINDOWS        = 50;
    const size_t     SAMPLES_PER_WINDOW = 5;
    const double     alpha              = 0.5;

    // -------- 4. 创建功耗采样器 --------
    PowerSampler sampler(
        /*period_ms=*/20,
        /*base=*/"/sys/class/power_supply/battery",
        /*tz_path=*/tz_path,
        /*log_path=*/""  // 不单独落盘 trace
    );

    // -------- 5. 打开 CSV --------
    ensure_window_csv_opened("/data/local/tmp/cpp/window_metrics.csv");

    // 启动采样线程（多轮算法复用一个 sampler）
    sampler.start();

    // -------- 6. 决定本轮要跑哪些优化器 --------
    std::vector<std::string> algo_list;
    if (algo_flag == "all") {
        // 只按你说的五个：grid / linear / neighbor / mab / bayes
        algo_list = { "grid", "linear", "neighbor", "mab", "bayes" };
    } else {
        algo_list = { algo_flag };  // 保留单算法模式
    }

    for (size_t i = 0; i < algo_list.size(); ++i) {
        const std::string & flag = algo_list[i];
        std::cout << "\n\n[EXEC] ====== Prepare to run algo " << flag << " (" << (i + 1) << "/" << algo_list.size()
                  << ") ======\n";

        // 每个算法开始前都等一次冷却
        wait_cool_down(tz_path, /*targetC=*/50.0, /*timeout_sec=*/300);

        run_static_for_algo(flag, runner, sampler, freqLevelsKHz, threadLevels, MAX_WINDOWS, SAMPLES_PER_WINDOW, alpha);
    }

    sampler.stop();

    // -------- 7. 收尾 --------
    if (g_windowCsv.is_open()) {
        g_windowCsv.close();
    }

    llama_backend_free();
    if (!write_wakelock("/sys/power/wake_unlock", "llmbench")) {
        std::fprintf(stderr, "return wakelock failed (need root/SELinux permissive)\n");
    }

    return 0;
}
