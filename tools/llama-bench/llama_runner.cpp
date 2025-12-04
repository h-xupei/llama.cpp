// llama_runner_single.cpp  （你也可以直接贴到 dynamic_opt_main.cpp 顶部）

#include "ggml.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// ================== 小工具函数：时间戳 ==================

static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

// ================== 小工具函数：预填充 ==================
// 基本逻辑参考 llama-bench 的 test_prompt，但完全独立实现

static bool run_prefill(llama_context * ctx, int n_prompt, int n_batch, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;

    while (n_processed < n_prompt) {
        int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0] = n_processed == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; ++i) {
            tokens[i] = std::rand() % n_vocab;
        }
        int res = llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens));
        if (res != 0) {
            std::fprintf(stderr, "%s: failed to decode prompt batch, res = %d\n", __func__, res);
            return false;
        }
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);
    return true;
}

// ================== 一次请求的指标结构 ==================

struct LlamaRunMetrics {
//    double energy;           // 本次请求能耗 (E_after - E_before)，单位 mJ（当前伪实现）
    double total_latency_s;  // 总延迟：从请求开始到最后一个生成 token 完成
    double ftl_s;            // First Token Latency：请求开始 -> 第一个生成 token 完成
    double steady_ts;        // 稳态 token/s（第二个生成 token 到最后一个）
    double overall_ts;       // 整体 token/s（含 prompt + gen）
};

// ================== LlamaRunner 定义 ==================

class LlamaRunner {
  public:
    // model_path: 模型 gguf 路径
    // n_ctx     : 上下文长度（>= 64+32）
    // n_batch   : batch size（可以用 2048）
    // n_threads : 初始解码线程数（后续可通过 set_num_threads 动态修改）
    LlamaRunner(const std::string & model_path, int n_ctx, int n_batch, int n_threads);
    ~LlamaRunner();

    // 跑一次 n_prompt + n_gen，返回各种指标
    LlamaRunMetrics run_one_request(int n_prompt_tokens = 64, int n_gen_tokens = 32);

    llama_context * ctx() const { return ctx_; }

    llama_model * model() const { return model_; }

    // 设置当前频率（单位：GHz），供伪能耗模型使用
    void set_freq_ghz(double f_ghz) { cur_freq_ghz_ = f_ghz; }

    // ===== 新增：设置推理线程数（在线调档时调用） =====
    void set_num_threads(int n_threads) {
        n_threads_ = n_threads;
        if (ctx_) {
            llama_set_n_threads(ctx_, n_threads_, n_threads_);
        }
    }

  private:
    // TODO: 用你真机的功耗采集改掉这里
    double read_energy();

  private:
    llama_model *   model_        = nullptr;
    llama_context * ctx_          = nullptr;
    int             n_batch_      = 0;
    int             n_threads_    = 0;
    double          cur_freq_ghz_ = 1.0;
};

// ================== LlamaRunner 实现 ==================

LlamaRunner::LlamaRunner(const std::string & model_path, int n_ctx, int n_batch, int n_threads) :
    model_(nullptr),
    ctx_(nullptr),
    n_batch_(n_batch),
    n_threads_(n_threads) {
    // 1. 模型参数（默认纯 CPU）
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;  // 纯 CPU；要 GPU 自己改
    mparams.use_mmap           = true;
    mparams.no_host            = false;

    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) {
        throw std::runtime_error("LlamaRunner: failed to load model: " + model_path);
    }

    // 2. 上下文参数
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = n_ctx;
    cparams.n_batch              = n_batch_;
    cparams.n_ubatch             = n_batch_;
    cparams.type_k               = GGML_TYPE_F16;
    cparams.type_v               = GGML_TYPE_F16;
    cparams.offload_kqv          = true;
    cparams.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.embeddings           = false;
    cparams.op_offload           = true;
    cparams.swa_full             = false;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        llama_model_free(model_);
        throw std::runtime_error("LlamaRunner: failed to create context");
    }

    // 初始化线程数
    llama_set_n_threads(ctx_, n_threads_, n_threads_);
}

LlamaRunner::~LlamaRunner() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

// 伪能耗模型：按时间积分，能耗随“有效频率”线性变化
double LlamaRunner::read_energy() {
    // 累积能量（单位：mJ）
    static double   E_acc_mJ  = 0.0;
    // 上一次调用 read_energy() 的时间
    static uint64_t last_t_ns = 0;

    uint64_t now_ns = get_time_ns();

    // 第一次调用：只初始化时间戳，不累积能量
    if (last_t_ns == 0) {
        last_t_ns = now_ns;
        return E_acc_mJ;
    }

    double dt_s = (now_ns - last_t_ns) * 1e-9;  // 时间间隔（秒）
    last_t_ns   = now_ns;

    // ===== 伪功耗模型 =====
    // 假想：P_total = P_static + k * f(GHz)
    // 这里纯拍脑袋：静态功耗 0.5 W，动态部分按频率线性增长
    const double P_static_mW      = 500.0;  // 0.5 W
    const double k_dyn_mW_per_GHz = 800.0;  // 每 1 GHz 增加 0.8 W
    double       f_ghz            = cur_freq_ghz_ > 0.0 ? cur_freq_ghz_ : 1.0;

    double P_mW = P_static_mW + k_dyn_mW_per_GHz * f_ghz;

    // 能量增量：E = P * t
    // mW * s = mJ
    double dE_mJ = P_mW * dt_s;

    E_acc_mJ += dE_mJ;

    return E_acc_mJ;
}

// 一次完整 64+32 推理 & 指标统计
LlamaRunMetrics LlamaRunner::run_one_request(int n_prompt_tokens, int n_gen_tokens) {
    if (!ctx_) {
        throw std::runtime_error("LlamaRunner::run_one_request: ctx_ is null");
    }

    // 清理内部 buffer
    llama_memory_clear(llama_get_memory(ctx_), false);

    // 按当前 n_threads_ 设置线程数（支持在线调档）
    llama_set_n_threads(ctx_, n_threads_, n_threads_);

    const llama_model * lmodel  = llama_get_model(ctx_);
    const llama_vocab * vocab   = llama_model_get_vocab(lmodel);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    // ===== 1. 起始能量 & 时间 =====
    double   E_before = read_energy();
    uint64_t t_start  = get_time_ns();

    // ===== 2. 预填充阶段 =====
    bool ok = run_prefill(ctx_, n_prompt_tokens, n_batch_, n_threads_);
    if (!ok) {
        throw std::runtime_error("LlamaRunner: prefill failed");
    }

    uint64_t t_prefill_done = get_time_ns();  // 如需单独统计 prefill 可用

    // ===== 3. 生成阶段：记录 FTL 与最后 token 时间 =====

    llama_token token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;

    uint64_t t_first_token_end = 0;
    uint64_t t_last_token_end  = 0;

    for (int i = 0; i < n_gen_tokens; ++i) {
        int res = llama_decode(ctx_, llama_batch_get_one(&token, 1));
        if (res != 0) {
            std::fprintf(stderr, "%s: failed to decode gen token, res = %d\n", __func__, res);
            throw std::runtime_error("LlamaRunner: gen failed");
        }
        llama_synchronize(ctx_);

        uint64_t t_now = get_time_ns();
        if (i == 0) {
            t_first_token_end = t_now;  // 首 token 完成时间
        }
        t_last_token_end = t_now;       // 不断刷新：最后 token 完成时间

        token = std::rand() % n_vocab;
    }

    // ===== 4. 结束能量 =====
    double E_after = read_energy();

    // ===== 5. 指标汇总 =====
    LlamaRunMetrics m{};

//    m.energy          = E_after - E_before;
    m.total_latency_s = (t_last_token_end - t_start) * 1e-9;   // 总延迟
    m.ftl_s           = (t_first_token_end - t_start) * 1e-9;  // FTL

    int total_tokens = n_prompt_tokens + n_gen_tokens;
    m.overall_ts     = total_tokens / m.total_latency_s;  // 含 prompt + gen

    if (n_gen_tokens > 1) {
        double steady_time   = (t_last_token_end - t_first_token_end) * 1e-9;
        int    steady_tokens = n_gen_tokens - 1;
        m.steady_ts          = steady_tokens / steady_time;  // 稳态 token/s
    } else {
        m.steady_ts = 0.0;
    }

    // 如果后面需要：double prefill_latency = (t_prefill_done - t_start) * 1e-9;

    return m;
}
