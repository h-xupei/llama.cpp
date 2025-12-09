// power_sampler.cpp

#pragma once

#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ================= 工具函数 =================

static inline uint64_t ps_now_ns() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
}

static inline bool ps_file_exists(const char * path) {
    struct stat st {};

    return ::stat(path, &st) == 0;
}

static inline bool ps_file_nonempty(const char * path) {
    struct stat st {};

    if (::stat(path, &st) != 0) {
        return false;
    }
    return S_ISREG(st.st_mode) && st.st_size > 0;
}

static inline bool ps_read_ll(const char * path, long long & out) {
    std::ifstream fin(path);
    if (!fin.good()) {
        return false;
    }
    long long v;
    fin >> v;
    if (fin.fail()) {
        return false;
    }
    out = v;
    return true;
}

// 把各种 sysfs 温度单位转成 deci-℃（0.1℃）
static inline bool ps_read_temp_to_dC(const char * path, long long & out_dC) {
    long long raw = 0;
    if (!ps_read_ll(path, raw)) {
        return false;
    }
    long long v = raw;

    // 粗略 heuristic：常见几种：
    //  - 毫℃:  35000 ~ 90000
    //  - 0.1℃: 350 ~ 900
    //  - ℃:   35 ~ 90
    if (std::llabs(v) > 10000) {
        // 毫℃ => 除以 100 得到 0.1℃
        v = v / 100;
    } else if (std::llabs(v) > 1000) {
        // 已经是 0.1℃，不变
    } else {
        // 当成 ℃，乘以 10
        v = v * 10;
    }

    out_dC = v;
    return true;
}

// ================= 采样点结构 =================

struct power_sample {
    uint64_t  ts_ns   = 0;    // 时间戳 (ns, steady clock)
    long long uV      = 0;    // 电压 (micro-Volt)
    long long uA      = 0;    // 电流 (micro-Ampere)
    double    mW      = 0.0;  // 功率 (milli-Watt)
    long long temp_dC = -1;   // 温度 (0.1 ℃)，<0 表示无效
};

// ================= PowerSampler 实现 =================

class PowerSampler {
  public:
    explicit PowerSampler(int         period_ms = 50,
                          std::string base      = "/sys/class/power_supply/battery",
                          std::string tz_path   = "/sys/class/thermal/thermal_zone37/temp",
                          std::string log_path  = "") :
        period_ms_(period_ms),
        base_(std::move(base)),
        tz_(std::move(tz_path)),
        log_path_(std::move(log_path)) {}

    ~PowerSampler() { stop(); }

    // 启动采样线程；可重复调用，running_ 为 true 时直接返回
    bool start() {
        if (running_) {
            return true;
        }

        v_path_ = base_ + "/voltage_now";
        i_path_ = base_ + "/current_now";
        t_path_ = tz_.empty() ? (base_ + "/temp") : tz_;

        if (!ps_file_exists(v_path_.c_str()) || !ps_file_exists(i_path_.c_str())) {
            std::fprintf(stderr, "PowerSampler: voltage_now/current_now not found under %s\n", base_.c_str());
            return false;
        }

        // 打开日志（若指定了路径）
        if (!log_path_.empty() && !log_fp_) {
            bool write_header = true;

            struct stat st {};

            if (::stat(log_path_.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
                write_header = false;
            }
            log_fp_ = std::fopen(log_path_.c_str(), "a");
            if (!log_fp_) {
                std::fprintf(stderr, "PowerSampler: cannot open log file %s\n", log_path_.c_str());
            } else if (write_header) {
                std::fprintf(log_fp_, "ts_ns,phase,voltage_uV,current_uA,power_mW,temp_deciC,event\n");
                std::fflush(log_fp_);
            }
        }

        running_ = true;
        th_      = std::thread([this] {
            prctl(PR_SET_NAME, "PowerSampler", 0, 0, 0);
            const size_t flush_every = 20;
            size_t       cnt         = 0;

            while (running_) {
                long long uV = 0, uA = 0, tdC = -1;
                uint64_t  ts  = ps_now_ns();
                bool      okV = ps_read_ll(v_path_.c_str(), uV);
                bool      okI = ps_read_ll(i_path_.c_str(), uA);

                if (ps_file_exists(t_path_.c_str())) {
                    long long dC = 0;
                    if (ps_read_temp_to_dC(t_path_.c_str(), dC)) {
                        tdC = dC;
                    }
                }

                if (okV && okI) {
                    power_sample s;
                    s.ts_ns   = ts;
                    s.uV      = uV;
                    s.uA      = uA;
                    s.temp_dC = tdC;
                    // μV * μA / 1e6 => mW
                    s.mW      = (double) uV * (double) std::abs(uA) / 1e6;

                    {
                        std::lock_guard<std::mutex> g(mu_);
                        buf_.push_back(s);
                        if (log_fp_) {
                            const char * phase = phase_.empty() ? "UNSET" : phase_.c_str();
                            std::fprintf(log_fp_, "%llu,%s,%lld,%lld,%.3f,%lld,\n", (unsigned long long) ts, phase, uV,
                                              uA, s.mW, tdC);
                            if (++cnt % flush_every == 0) {
                                std::fflush(log_fp_);
                            }
                        }
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(std::max(1, period_ms_)));
            }
        });

        return true;
    }

    void stop() {
        if (!running_) {
            return;
        }
        running_ = false;
        if (th_.joinable()) {
            th_.join();
        }
        if (log_fp_) {
            std::fflush(log_fp_);
            std::fclose(log_fp_);
            log_fp_ = nullptr;
        }
    }

    // 标记阶段名；后续采样行会带上 phase 字段
    void mark_stage(const std::string & phase_name) {
        std::lock_guard<std::mutex> g(mu_);
        phase_ = phase_name;
        if (log_fp_) {
            uint64_t ts = ps_now_ns();
            std::fprintf(log_fp_, "%llu,%s,,,,,MARK\n", (unsigned long long) ts, phase_.c_str());
            std::fflush(log_fp_);
        }
    }

    // 拿一份只读快照，用于积分/统计
    std::vector<power_sample> snapshot() const {
        std::lock_guard<std::mutex> g(mu_);
        return buf_;
    }

    // ===== 虚假功率和能量函数（基于频率、线程数和窗口时间） =====

    /**
 * 计算虚假平均功率，基于CPU频率和线程数
 *
 * 函数形式：P_fake(f, n) = 4000 × (f/2.4192)^0.8 × (n/3)^0.7 × (1 + 0.3×(1 - f/2.4192))
 *
 * 这个函数模拟的物理现象：
 *   1. 频率越高，功率越大（正相关，指数0.8）
 *   2. 线程数越多，功率越大（正相关，指数0.7）
 *   3. 在低频时，由于效率下降，有额外的功率惩罚（+30%）
 *
 * @param f_khz CPU频率，单位kHz
 * @param n_threads 线程数 (1, 2, 3)
 * @return 虚假平均功率，单位mW
 */
    static double fake_power_mW(double f_khz, int n_threads) {
        if (f_khz <= 0.0 || n_threads <= 0) {
            return 0.0;
        }

        double f_ghz = f_khz / 1e6;

        const double P_static = 800.0;   // 静态 + 其它模块底噪 (mW)
        const double P_max    = 3200.0;  // 纯动态功耗在最高档时的增量 (mW)
        const double f_max    = 2.4192;  // GHz
        const double n_max    = 3.0;
        const double gamma    = 0.8;
        const double delta    = 0.7;
        const double alpha    = 0.3;

        double f_norm = std::clamp(f_ghz / f_max, 0.01, 1.0);
        double n_norm = std::clamp((double)n_threads / n_max, 0.01, 1.0);

        double dyn = P_max
                     * std::pow(f_norm, gamma)
                     * std::pow(n_norm, delta)
                     * (1.0 + alpha * (1.0 - f_norm));

        return P_static + dyn;
    }


    /**
 * 计算虚假总能量，基于功率和窗口时间
 *
 * 能量 = 功率 × 时间 × 1000 (mW × s → mJ)
 *
 * @param f_khz CPU频率，单位kHz
 * @param n_threads 线程数 (1, 2, 3)
 * @param window_time_s 窗口持续时间，单位秒
 * @return 虚假总能量，单位mJ
 */
    static double fake_energy_mJ(double f_khz, int n_threads, double window_time_s) {
        if (window_time_s <= 0.0) {
            return 0.0;
        }

        // 计算平均功率
        double avg_power_mW = fake_power_mW(f_khz, n_threads);

        // 能量 = 功率 × 时间 × 1000 (转换为mJ)
        // 注意：1 mW × 1 s = 1 mJ
        double energy_mJ = avg_power_mW * window_time_s;

        return energy_mJ;
    }

    // ===== 静态分析函数：积分能量 / 平均功率 / 温度等 =====

    // 对 [t0, t1] 上的功率曲线做梯形积分，返回能量 (mJ)
    static double integrate_mJ(const std::vector<power_sample> & s, uint64_t t0, uint64_t t1) {
        if (s.size() < 2 || t1 <= t0) {
            return 0.0;
        }
        double mJ = 0.0;
        size_t i  = 0;

        // 跳过早于 t0 的点
        while (i + 1 < s.size() && s[i + 1].ts_ns < t0) {
            ++i;
        }

        for (; i + 1 < s.size(); ++i) {
            uint64_t a = std::max<uint64_t>(s[i].ts_ns, t0);
            uint64_t b = std::min<uint64_t>(s[i + 1].ts_ns, t1);
            if (b <= a) {
                continue;
            }

            double Pa   = s[i].mW;
            double Pb   = s[i + 1].mW;
            double dt_s = (double) (b - a) / 1e9;
            mJ += 0.5 * (Pa + Pb) * dt_s;
        }
        return mJ;
    }

    static double first_power_after(const std::vector<power_sample> & s, uint64_t t0) {
        if (s.empty()) {
            return 0.0;
        }

        for (const auto & x : s) {
            if (x.ts_ns < t0) {
                continue;
            }
            if (x.temp_dC < 0) {
                continue;  // 无效温度
            }
            return static_cast<double>(x.mW);
        }
        return 0.0;
    }

    static double avg_mW(const std::vector<power_sample> & s, uint64_t t0, uint64_t t1) {
        if (t1 <= t0) {
            return 0.0;
        }
        double mJ   = integrate_mJ(s, t0, t1);
        double dt_s = (double) (t1 - t0) / 1e9;
        return dt_s > 0 ? (mJ / dt_s) : 0.0;
    }

    static double avg_temp_dC(const std::vector<power_sample> & s, uint64_t t0, uint64_t t1) {
        if (s.size() < 2 || t1 <= t0) {
            return 0.0;
        }
        double sum = 0.0, wsum = 0.0;
        size_t i = 0;

        while (i + 1 < s.size() && s[i + 1].ts_ns < t0) {
            ++i;
        }

        for (; i + 1 < s.size(); ++i) {
            uint64_t a = std::max<uint64_t>(s[i].ts_ns, t0);
            uint64_t b = std::min<uint64_t>(s[i + 1].ts_ns, t1);
            if (b <= a) {
                continue;
            }

            double w  = (double) (b - a);
            double Ta = s[i].temp_dC;
            double Tb = s[i + 1].temp_dC;
            if (Ta < 0 || Tb < 0) {
                continue;  // 没有温度就跳过
            }
            sum += 0.5 * (Ta + Tb) * w;
            wsum += w;
        }
        return wsum > 0 ? (sum / wsum) : 0.0;
    }

    static double max_temp_dC(const std::vector<power_sample> & s, uint64_t t0, uint64_t t1) {
        if (s.empty() || t1 <= t0) {
            return 0.0;
        }
        long long mx = -1000000000;
        for (const auto & x : s) {
            if (x.ts_ns < t0 || x.ts_ns > t1) {
                continue;
            }
            if (x.temp_dC > mx) {
                mx = x.temp_dC;
            }
        }
        return mx < 0 ? 0.0 : (double) mx;
    }

    // 返回 t0 之后（含 t0）的第一个有效温度，单位：deci-℃（0.1℃）
    // 如果找不到有效温度，返回 0.0
    static double first_temp_dC_after(const std::vector<power_sample> & s, uint64_t t0) {
        if (s.empty()) {
            return 0.0;
        }

        for (const auto & x : s) {
            if (x.ts_ns < t0) {
                continue;
            }
            if (x.temp_dC < 0) {
                continue;  // 无效温度
            }
            return static_cast<double>(x.temp_dC);
        }
        return 0.0;
    }

    static void dump_csv(const std::vector<power_sample> & s, const std::string & path) {
        if (path.empty()) {
            return;
        }
        std::ofstream out(path);
        if (!out.good()) {
            return;
        }
        out << "ts_ns,voltage_uV,current_uA,power_mW,temp_deciC\n";
        for (const auto & x : s) {
            out << x.ts_ns << "," << x.uV << "," << x.uA << "," << x.mW << "," << x.temp_dC << "\n";
        }
    }

    // 轻量级追加 CSV，只写 exp_id/phase/I/V/结束时间，方便后处理
    static void append_power_samples_csv(const std::vector<power_sample> & snap,
                                         uint64_t                          t0,
                                         uint64_t                          t1,
                                         const char *                      phase,
                                         const std::string &               exp_id,
                                         const std::string &               sys_time_end,
                                         const char * path = "/data/local/tmp/cpp/power_samples.csv") {
        if (t1 <= t0) {
            return;
        }

        bool   has_file = ps_file_nonempty(path);
        FILE * fp       = std::fopen(path, has_file ? "a" : "w");
        if (!fp) {
            return;
        }

        if (!has_file) {
            std::fprintf(fp, "exp_id,phase,current_uA,voltage_uV,sys_time_end\n");
        }

        for (const auto & s : snap) {
            if (s.ts_ns < t0 || s.ts_ns > t1) {
                continue;
            }
            std::fprintf(fp, "%s,%s,%lld,%lld,%s\n", exp_id.c_str(), phase ? phase : "UNSET", s.uA, s.uV,
                         sys_time_end.c_str());
        }
        std::fclose(fp);
    }

  private:
    int         period_ms_;
    std::string base_, tz_;
    std::string v_path_, i_path_, t_path_;

    mutable std::mutex        mu_;
    std::vector<power_sample> buf_;
    std::atomic<bool>         running_{ false };
    std::thread               th_;

    // 日志相关
    std::string log_path_;
    FILE *      log_fp_ = nullptr;
    std::string phase_;
};
