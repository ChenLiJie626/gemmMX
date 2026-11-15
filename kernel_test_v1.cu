#include <mc_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <chrono>
#include "mctlass/arch/maca_memory.h" //MACA_ICMP_SLT 等


#define arrive_gvmcnt(count) __builtin_mxc_arrive(64 + count);
#define cp_async_fenc() asm(";--------------")


#define CHECK_DATA_PATH 0

namespace gemm_config
{
    static constexpr int kTileM = 128;
    static constexpr int kTileN = 128;
    static constexpr int kTileK = 256;

    static constexpr int kStage      = 4;       // 当前 wave 的 M/N 方向各有四组 16 行输入片段
    static constexpr int kThreadNum  = 256;     // block 中共有 256 个线程
    static constexpr int kWarpSize   = 64;
    static constexpr int kWaveNum    = kThreadNum / kWarpSize; // 4
    static constexpr int kWaveM      = 2;
    static constexpr int kWaveN      = kWaveNum / kWaveM;      // 2

    static constexpr int kABSize = kTileK * kTileN;            // M==N, A/B 各一个 tile 元素数 = 32768
    static constexpr int kLdgBytes = 16;
    static constexpr int kElemsPerLdg = kLdgBytes / (int)sizeof(int8_t);   // 每个线程一次搬 16 个 INT8，共 16B

    using LdgType   = __NATIVE_VECTOR__(4, int32_t);   // 16B
    using LdsType   = LdgType;
    using RegType   = int32_t;
    using ABType    = int32_t;
    using AccumType = __NATIVE_VECTOR__(4, int32_t);

    static constexpr int kLdgThreadMN = 4;
    static constexpr int kLdgThreadK  = 16;  //安排 16 个线程合作搬一行
    static constexpr int kLdgSize        = sizeof(LdgType) * kThreadNum;      // 16*256 = 4096
    static constexpr int kLdgSizePerWave = kLdgSize / kWaveNum;               // 1024
    static constexpr int kLdgNum         = kABSize * (int)sizeof(int8_t) / kLdgSize; // 8
    static constexpr int kLdgNumPerStage = kLdgNum / kStage;                  // 2
    static constexpr int kLdgMNStride    = kTileN / kLdgNumPerStage;          // 64 行跨度

    static constexpr int kMmaThreadMN = 16;    // 一个 MMA 片段在 M/N 方向的大小
    static constexpr int kMmaThreadK  = 4;
    static constexpr int kLdsNumPerK  = kTileK / (int)sizeof(LdsType) / kMmaThreadK; // 4
    static constexpr int kLdsRowStride = kMmaThreadMN * kWaveM;                  // 32
    static constexpr int kLdsColStride = kMmaThreadMN * kWaveN;                  // 32

    static constexpr int MMA_M = kTileM / 16 / kWaveM;   // 4
    static constexpr int MMA_N = kTileN / 16 / kWaveN;   // 4

    // smem: 仅一个窗口 A+B = 64KB (stage 在寄存器)
    static constexpr size_t kSMEMA  = kTileM * kTileK * sizeof(int8_t); // 32768
    static constexpr size_t kSMEMB  = kTileN * kTileK * sizeof(int8_t); // 32768
    static constexpr size_t kSMEMTotal = kSMEMA + kSMEMB;

    using ElementC = int32_t;
}

// ---- 16B 向量 LDS/STG 宏 ----
#define LDS_OFS(dst, src, ofs)                                                        \
    do {                                                                              \
        cp_async_fenc();                                                              \
        *reinterpret_cast<gemm_config::LdsType *>(&(dst)) =                           \
            *reinterpret_cast<gemm_config::LdsType *>(&(src) + (ofs));                \
        cp_async_fenc();                                                              \
    } while (0)

#define MMA_MNK_I(mm, nn, kk, ii)                                                       \
    acc[mm][nn] = __builtin_mxc_mma_16x16x16i8(a[mm][kk][(ii) * 2],     b[nn][kk][(ii) * 2],     acc[mm][nn]); \
    acc[mm][nn] = __builtin_mxc_mma_16x16x16i8(a[mm][kk][(ii) * 2 + 1], b[nn][kk][(ii) * 2 + 1], acc[mm][nn]);

// host 侧 A: MxK row-major, B: row-major, C: MxN
__global__ void int8_gemm_128x128x256_v3(
    const int8_t *__restrict__ A,
    const int8_t *__restrict__ B,
    int32_t *__restrict__ C,
    int M, int N, int K)
{
    using namespace gemm_config;

    int tid  = threadIdx.x;
    int wave = tid / kWarpSize;   // 0..3  (wave_m = wave/2, wave_n = wave%2)
    int lane = tid % kWarpSize;

    const int block_row = blockIdx.y * kTileM;
    const int block_col = blockIdx.x * kTileN;

    const int wave_m = wave / 2;
    const int wave_n = wave % 2;

    const int MMA_K = 16;

    extern __shared__ int8_t smem[];
    int8_t *smem_A = smem;
    int8_t *smem_B = smem + kSMEMA;

    // ---- 读侧(LDS) 坐标 (标准 XOR) ----
    int lds_mn      = tid % kMmaThreadMN;
    int lds_m_base  = lds_mn + (wave / 2) * kLdgMNStride;
    int lds_n_base  = lds_mn + (wave % 2) * kLdgMNStride;
    int asld[kLdsNumPerK], bsld[kLdsNumPerK];
    #pragma unroll
    for (int i = 0; i < kLdsNumPerK; ++i)
    {
        int lds_k = (lane / kMmaThreadMN) * kElemsPerLdg + i * kLdgMNStride;
        asld[i] = lds_m_base * kTileK + lds_k;
        bsld[i] = lds_n_base * kTileK + lds_k;
    }

    ABType a[kStage][kLdsNumPerK][4];
    ABType b[kStage][kLdsNumPerK][4];
    AccumType acc[MMA_M][MMA_N];

    #pragma unroll
    for (int m = 0; m < MMA_M; ++m)
        #pragma unroll
        for (int n = 0; n < MMA_N; ++n)
            acc[m][n] = (AccumType){0, 0, 0, 0};

    int num_tile_k = K / kTileK;   // K 窗口数(K 须 256 倍数)

    for (int tilek = 0; tilek < num_tile_k; ++tilek)
    {
        
        for (int i = 0; i < kTileM / (kThreadNum / kLdgThreadK); ++i) {   // 这里要搬运8轮
            int rowA_s = tid / kLdgThreadK + i * (kThreadNum / kLdgThreadK);
            int colA_s = (tid % kLdgThreadK) * kElemsPerLdg;
            int rowB_s = tid / kLdgThreadK + i * (kThreadNum / kLdgThreadK);
            int colB_s = (tid % kLdgThreadK) * kElemsPerLdg;

            int rowA_g = block_row + rowA_s;
            int colA_g = tilek * kTileK + colA_s;
            int rowB_g = block_col + rowB_s;
            int colB_g = tilek * kTileK + colB_s;

            __builtin_mxc_ldg_b128_bsm_predicator(
                smem_A + rowA_s * kTileK + colA_s,
                (void *)&A[rowA_g * K + colA_g],
                0, true, true, false, true, colA_g, K, MACA_ICMP_SLT);
            __builtin_mxc_ldg_b128_bsm_predicator(
                smem_B + rowB_s * kTileK + colB_s,
                (void *)&B[rowB_g * K + colB_g],
                0, true, true, false, true, colB_g, K, MACA_ICMP_SLT);
        }
        arrive_gvmcnt(0);
        __builtin_mxc_barrier_inst();

        #pragma unroll
        for (int s = 0; s < kStage; ++s) {
            #pragma unroll
            for (int k = 0; k < kLdsNumPerK; ++k) {
                LDS_OFS(a[s][k][0], smem_A[asld[k]], s * kMmaThreadMN * kTileK);
                LDS_OFS(b[s][k][0], smem_B[bsld[k]], s * kMmaThreadMN * kTileK);
            }
        }

        for (int m = 0; m < MMA_M; ++m) {
            for (int n = 0; n < MMA_N; ++n) {
                for (int k = 0; k < kLdsNumPerK; ++k) {

                    MMA_MNK_I(m, n, k, 0);

                    MMA_MNK_I(m, n, k, 1);

                }
            }
        }
        
    }

    #pragma unroll
    for (int mm = 0; mm < MMA_M; ++mm)
    {
        #pragma unroll
        for (int nn = 0; nn < MMA_N; ++nn)
        {
            int mma_row = (lane / MMA_K) * 4;
            int row = block_row + wave_m * kWarpSize + mm * MMA_K + mma_row;
            int mma_col = lane % MMA_K;
            int col = block_col + wave_n * kWarpSize + nn * MMA_K + mma_col;
            C[(row + 0) * N + col] = acc[mm][nn][0];
            C[(row + 1) * N + col] = acc[mm][nn][1];
            C[(row + 2) * N + col] = acc[mm][nn][2];
            C[(row + 3) * N + col] = acc[mm][nn][3];
        }
    }
    
}


void verifyResult(const int8_t* A, const int8_t* B, const int32_t* C, int M, int N, int K)
{
    long long total_ops = (long long)M * N * K;
    const long long FULL_LIMIT = 200000000LL;
    int errors = 0;
    long long max_diff = 0;

    if (total_ops <= FULL_LIMIT) {
        printf("\n=== 验证结果 (全量 %dx%d, K=%d) ===\n", M, N, K);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                long long ref = 0;
                for (int k = 0; k < K; ++k)
                    ref += (long long)A[i * K + k] * (long long)B[j * K + k];
                int32_t got = C[i * N + j];
                long long diff = ref > (long long)got ? ref - (long long)got : (long long) got - ref;
                if (diff != 0) { errors++; if (diff > max_diff) max_diff = diff; }
            }
        if (errors == 0) printf("√ 全部 %d 个元素正确\n", M * N);
        else             printf("× 发现 %d 个错误, 最大差值=%lld\n", errors, max_diff);
    } else {
        const long long S = 300000;
        printf("\n=== 验证结果 (抽样 %lld / %d 个元素, K=%d) ===\n", S, M * N, K, total_ops);
        unsigned int seed = 987654321u;
        auto rnd = [&seed](unsigned int mod) {
            seed = seed * 1103515245u + 12345u;
            return (int)((seed >> 8) % mod);
        };
        long long checked = 0;
        for (long long s = 0;s < S; ++s) {
            int i = rnd((unsigned int)M);
            int j = rnd((unsigned int)N);
            long long ref = 0;
            for (int k = 0; k < K; ++k)
                ref += (long long)A[i * K + k] * (long long)B[j * K + k];
            int32_t got = C[i * N + j];
            long long diff = ref > (long long)got ? ref - (long long)got : (long long)got - ref;
            if (diff != 0) { errors++; if (diff > max_diff) max_diff = diff; }
            ++checked;
        }
        if (errors == 0) printf("√ 抽样 %lld 个元素全部正确 (占 %.4f%%)\n",
                                checked, (double)checked * 100.0 / ((double)M * N));
        else             printf("× 抽样发现 %d 个错误, 最大差值=%lld\n", errors, max_diff);
    }
}

void fill_input(int8_t* h_A, int8_t* h_B, int M, int N, int K, int mode)
{
    if (mode == 0) {
        for (int r = 0; r < M; ++r)
            for (int k = 0; k < K; ++k) h_A[r * K + k] = (int8_t)(r + k);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) h_B[n * K + k] = (int8_t)(k + n);
    } else if (mode == 1) {
        for (int i = 0; i < M * K; ++i) h_A[i] = 1;
        for (int i = 0; i < K * M; ++i) h_B[i] = 1;
    } else {
        unsigned int seed = 12345u + mode * 7u;
        auto rnd = [&seed]() {
            seed = seed * 1103515245u + 12345u;
            return (int8_t)((seed >> 16) & 0xFF);
        };
        for (int i = 0; i < M * K; ++i) h_A[i] = rnd();
        for (int i = 0; i < K * N; ++i) h_B[i] = rnd();
    }
}

void print_5x5(const int8_t* h_A, const int8_t* h_B, const int32_t* h_C, int M, int N, int K)
{
    const int R = M < 5 ? M : 5;
    const int C = N < 5 ? N : 5;
    printf("   左上 %dx%d (实际 / 期望):\n", R, C);
    for (int i = 0; i < R; ++i) {
        printf("   行%d 实际:", i);
        for (int j = 0; j < C; ++j) printf(" %6d", h_C[i * N + j]);
        printf("\n");
        printf("       期望:");
        for (int j = 0; j < C; ++j) {
            long long ref = 0;
            for (int k = 0; k < K; ++k)
                ref += (long long)h_A[i * K + k] * (long long)h_B[j * K + k];
            printf(" %6lld", ref);
        }
        printf("\n");
    }
}

void run_case(int id, int mode, const char* name, int M, int N, int K)
{
    using namespace gemm_config;
    printf("\n########## Case %d: %s  [M=%d N=%d K=%d] ##########\n", id, name, M, N, K);

    int8_t* h_A = new int8_t[M * K];
    int8_t* h_B = new int8_t[K * N];
    int32_t* h_C = new int32_t[M * N];

    fill_input(h_A, h_B, M, N, K, mode);

    int8_t *d_A, *d_B;
    int32_t *d_C;
    mcMalloc((void**)&d_A, M * K * sizeof(int8_t));
    mcMalloc((void**)&d_B, K * N * sizeof(int8_t));
    mcMalloc((void**)&d_C, M * N * sizeof(int32_t));
    mcMemcpy(d_A, h_A, M * K * sizeof(int8_t), mcMemcpyHostToDevice);
    mcMemcpy(d_B, h_B, K * N * sizeof(int8_t), mcMemcpyHostToDevice);
    mcMemset(d_C, 0, M * N * sizeof(int32_t));

    dim3 grid(N / kTileN, M / kTileM, 1);
    dim3 block(kThreadNum, 1, 1);

    auto t0 = std::chrono::steady_clock::now();
    int8_gemm_128x128x256_v3<<<grid, block, kSMEMTotal>>>(d_A, d_B, d_C, M, N, K);
    mcDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    mcError_t err = mcGetLastError();
    if (err != mcSuccess) {
        printf("× Kernel执行失败: %s\n", mcGetErrorString(err));
        delete[] h_A; delete[] h_B; delete[] h_C;
        mcFree(d_A); mcFree(d_B); mcFree(d_C);
        return;
    }

    mcMemcpy(h_C, d_C, M * N * sizeof(int32_t), mcMemcpyDeviceToHost);
    verifyResult(h_A, h_B, h_C, M, N, K);
    print_5x5(h_A, h_B, h_C, M, N, K);
    printf("   kernel 耗时: %.1f ms  (grid %dx%d = %d block)\n", ms, grid.x, grid.y, grid.x * grid.y);

    mcFree(d_A); mcFree(d_B); mcFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}

int main(int argc, char** argv)
{
    using namespace gemm_config;
    printf("tile 128x128x256, wave 2x2(64x64), kStage=4 M-band 流水(单窗口 smem 64KB)\n");
    printf("约束: M/N 为 %d 倍数, K 为 %d 倍数\n", kTileM, kTileK);

    struct Case
    {
        int mode;
        const char *name;
        int M, N, K;
    };
    Case cases[] = {
        {0,  "位置编码 A=r+k, B=n+k", 128, 128, 256},
        {1,  "全1 (C 全=K)",          128, 128, 256},
        {2,  "随机 int8",             128, 128, 256},
        {3,  "随机 128x128x512",      128, 128, 512},
        {4,  "随机 128x128x1024",     128, 128, 1024},
        {5,  "随机 256x256x256",      256, 256, 256},
        {6,  "随机 256x128x256",      256, 128, 256},
        {7,  "随机 128x256x256",      128, 256, 256},
        {8,  "随机 256x256x512",      256, 256, 512},
        {9,  "随机 4096x4096x256",    4096, 4096, 256},
        {10, "随机 4096x4096x512",    4096, 4096, 512},
        {11, "随机 4096x4096x2048",   4096, 4096, 2048},
    };
    const int NC = sizeof(cases) / sizeof(cases[0]);

    if (argc <= 1) {
        printf("将运行 case 0~2 (默认: 位置编码 / 全1 / 随机 128³)。\n\n");
        for (int i = 0; i <= 2; ++i)
            run_case(i, cases[i].mode, cases[i].name, cases[i].M, cases[i].N, cases[i].K);
        printf("\n完成。\n");
    } else {
        printf("将运行指定 case: ");
        for (int a = 1; a < argc; ++a) printf("%s ", argv[a]);
        printf("\n\n");
        for (int a = 1; a < argc; ++a) {
            int i = atoi(argv[a]);
            if (i < 0 || i >= NC) { printf("× case %d 越界 (0~%d)\n", i, NC - 1); continue; }
            run_case(i, cases[i].mode, cases[i].name, cases[i].M, cases[i].N, cases[i].K);
        }
        printf("\n指定 case 完成。\n");
    }
    return 0;
}