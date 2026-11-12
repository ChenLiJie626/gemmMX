// C500 FP16 x FP16 -> FP32 GEMM tutorial and benchmark.
// See README_逐段讲解.md for the native MMA/BSM contract and limitations.
// Intrinsic calling conventions informed by QingCheng-AI/muxi_native_layout_kernels
// (Apache-2.0); see NOTICE.md. No CUDA WMMA/PTX instructions are used.

// [01] MACA native runtime, not NVIDIA nvcc.
#include <maca.h>
#include <maca_fp16.h>
#include <mc_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using U2 = __NATIVE_VECTOR__(2, unsigned int);
using U4 = __NATIVE_VECTOR__(4, unsigned int);
using F4 = __NATIVE_VECTOR__(4, float);
constexpr int Wave = 64, TileM = 128, TileN = 128, TileK = 256, SliceK = 32;
static_assert(sizeof(__half) == 2 && sizeof(U4) == 16, "Unexpected MACA ABI");

#define MC_CHECK(expr) do { \
    mcError_t e = (expr); \
    if (e != mcSuccess) { \
        std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr, mcGetErrorString(e)); \
        std::exit(2); \
    } \
} while (0)

// [02] Each wave owns a 64x64 output. A lane holds 4x4 F4 = 64 FP32 sums.
struct Fragments { U4 a[4], b[4]; };
struct Accumulators { F4 c[4][4]; };

__device__ __forceinline__ U4 pack8(const __half* p) {
    // Deliberately scalar source loads for the introductory implementation.
    U4 v{};
    const uint16_t* bits = reinterpret_cast<const uint16_t*>(p);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        v[j] = unsigned(bits[2*j]) | (unsigned(bits[2*j+1]) << 16);
    return v;
}

// [03] Native MMA: first argument is mathematical B for our row-major output.
__device__ __forceinline__ void mma_half(Accumulators& acc,
                                        const Fragments& f, int half) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            U2 b{f.b[j][2*half], f.b[j][2*half+1]};
            U2 a{f.a[i][2*half], f.a[i][2*half+1]};
            acc.c[i][j] = __builtin_mxc_mma_16x16x16f16(b, a, acc.c[i][j]);
        }
    }
}

// [04] One wave: r = lane%16; q = lane/16.
// A fragment row=4*r+i, B fragment column=4*r+j, both use K=8*q..8*q+7.
__device__ __forceinline__ Fragments global_fragments(
    const __half* A, const __half* B, int K, int m0, int n0, int k0) {
    int lane = threadIdx.x & 63, r = lane & 15, q = lane >> 4;
    Fragments f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        f.a[j] = pack8(A + size_t(m0 + 4*r + j)*K + k0 + 8*q);
        f.b[j] = pack8(B + size_t(n0 + 4*r + j)*K + k0 + 8*q);
    }
    return f;
}

// [05] Shared layout in units of 16 bytes. Groups contain four rows x 32 K.
// Swizzle is a bijection within each group of sixteen 128-bit vectors.
template<bool Swizzle>
__device__ __forceinline__ int shared_index(int row, int kpack) {
    int group = row >> 2;
    int col = ((row & 3) << 2) | kpack;
    if constexpr (Swizzle) col ^= group & 15;
    return group*16 + col;
}

// [06] Real asynchronous global -> BSM intrinsic. saddr is WAVE-UNIFORM.
// In the referenced C500 ABI each lane writes at saddr + lane*16 bytes.
// Adding lane*16 again here would corrupt the layout.
__device__ __forceinline__ void copy_bsm(unsigned char* dst, const __half* src) {
    __builtin_mxc_ldg_b128_bsm_predicator(
        dst, src, 0, true, true, false, true, 1, 1, MACA_ICMP_EQ);
}

template<int BM, bool Swizzle, bool Builtin>
__device__ __forceinline__ void issue_stage(
    unsigned char* stage, const __half* A, const __half* B,
    int K, int m0, int n0, int k0) {
    constexpr int Threads = (BM/64)*(BM/64)*64;
    constexpr int Vectors = BM*4, MatrixBytes = Vectors*16;
    int tid = threadIdx.x;
    int wave = __builtin_mxc_readfirstlane(tid / 64);
    #pragma unroll
    for (int h = 0; h < Vectors/Threads; ++h) {
        int x = h*Threads + tid, group = x >> 4, col = x & 15;
        if constexpr (Swizzle) col ^= group & 15; // inverse XOR
        int row = group*4 + (col >> 2), kp = col & 3;
        const __half* ap = A + size_t(m0+row)*K + k0 + kp*8;
        const __half* bp = B + size_t(n0+row)*K + k0 + kp*8;
        if constexpr (Builtin) {
            int uniform_offset = (h*Threads + wave*64)*16;
            copy_bsm(stage + uniform_offset, ap);
            copy_bsm(stage + MatrixBytes + uniform_offset, bp);
        } else {
            // Teaching baseline: scalar global -> registers -> shared.
            // Compiler optimization can combine these loads/stores; inspect ISA.
            uint16_t* sa = reinterpret_cast<uint16_t*>(stage) + x*8;
            uint16_t* sb = reinterpret_cast<uint16_t*>(stage + MatrixBytes) + x*8;
            const uint16_t* ga = reinterpret_cast<const uint16_t*>(ap);
            const uint16_t* gb = reinterpret_cast<const uint16_t*>(bp);
            #pragma unroll
            for (int j = 0; j < 8; ++j) { sa[j] = ga[j]; sb[j] = gb[j]; }
        }
    }
}

// [07] Shared -> registers in the exact same native fragment layout as [04].
template<int BM, bool Swizzle>
__device__ __forceinline__ Fragments shared_fragments(
    const unsigned char* stage, int wm, int wn) {
    int lane = threadIdx.x & 63, r = lane & 15, q = lane >> 4;
    const U4* sa = reinterpret_cast<const U4*>(stage);
    const U4* sb = sa + BM*4;
    Fragments f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        f.a[j] = sa[shared_index<Swizzle>(wm*64 + 4*r+j, q)];
        f.b[j] = sb[shared_index<Swizzle>(wn*64 + 4*r+j, q)];
    }
    return f;
}

// [08] gvmcnt requires immediate thresholds: emit a finite set, not runtime N.
template<int Ops>
__device__ __forceinline__ void wait_stage(int newer_stages) {
    switch (newer_stages) {
        case 3: __builtin_mxc_arrive_gvmcnt(3*Ops); break;
        case 2: __builtin_mxc_arrive_gvmcnt(2*Ops); break;
        case 1: __builtin_mxc_arrive_gvmcnt(Ops); break;
        default: __builtin_mxc_arrive_gvmcnt(0); break;
    }
    // A counter wait belongs to this wave; the barrier joins all producer waves.
    __builtin_mxc_barrier_inst();
}

// [09] Output lane mapping is DIFFERENT from the input fragment mapping.
// acc[i][j][z] -> D[m0+4*r+i, n0+16*q+4*z+j].
__device__ __forceinline__ void epilogue(
    const Accumulators& acc, const float* C, float* D,
    int N, int m0, int n0, float alpha, float beta) {
    int lane = threadIdx.x & 63, r = lane & 15, q = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int z = 0; z < 4; ++z) {
                size_t index = size_t(m0+4*r+i)*N + n0+16*q+4*z+j;
                float value = alpha * acc.c[i][j][z];
                if (beta != 0.0f) value = fmaf(beta, C[index], value);
                D[index] = value;
            }
        }
    }
}

// [10] ONE CTA produces a 128x128 output tile; the logical K tile is 256.
template<int BM>
__device__ __forceinline__ void block_origin(int& m0, int& n0) {
    static_assert(BM == TileM && TileM == TileN, "This tutorial uses a 128x128 CTA");
    m0 = int(blockIdx.y)*TileM;
    n0 = int(blockIdx.x)*TileN;
}

template<int BM>
__global__ void direct_kernel(const __half* A, const __half* B,
    const float* C, float* D, int N, int K, float alpha, float beta) {
    int m0, n0; block_origin<BM>(m0, n0);
    int w = threadIdx.x/64, wm = w/(BM/64), wn = w%(BM/64);
    Accumulators acc{};
    // Explicit 256-wide logical reduction tiles; stream eight 32-wide slices.
    for (int outer = 0; outer < K; outer += TileK) {
        #pragma unroll 1
        for (int inner = 0; inner < TileK; inner += SliceK) {
            Fragments f = global_fragments(A, B, K, m0+64*wm, n0+64*wn, outer+inner);
            mma_half(acc, f, 0);
            mma_half(acc, f, 1);
        }
    }
    epilogue(acc, C, D, N, m0+64*wm, n0+64*wn, alpha, beta);
}

// [11] Circular multi-stage BSM pipeline; no full 256-K A/B tile in BSM.
template<int BM, int Stages, bool Swizzle, bool Builtin, bool RegPrefetch>
__global__ __launch_bounds__((BM/64)*(BM/64)*64)
void staged_kernel(const __half* A, const __half* B,
    const float* C, float* D, int N, int K, float alpha, float beta) {
    constexpr int Threads = (BM/64)*(BM/64)*64;
    constexpr int StageBytes = 2*BM*SliceK*int(sizeof(__half));
    constexpr int Ops = 2*(BM*4/Threads); // LDG instructions per wave per stage
    static_assert(Stages*StageBytes <= 65536, "BSM budget exceeded");
    static_assert(!RegPrefetch || (Builtin && Stages >= 2), "Invalid prefetch configuration");
    __shared__ __align__(16) unsigned char storage[Stages*StageBytes];
    int m0, n0; block_origin<BM>(m0, n0);
    int w = threadIdx.x/64, wm = w/(BM/64), wn = w%(BM/64);
    int tiles = K/SliceK;
    Accumulators acc{};

    if constexpr (!RegPrefetch) {
        // Prologue: leave one empty slot to issue the farthest future slice.
        constexpr int Initial = Stages == 1 ? 0 : Stages-1;
        #pragma unroll
        for (int s = 0; s < Initial; ++s)
            issue_stage<BM,Swizzle,Builtin>(storage+s*StageBytes, A,B,K,m0,n0,s*SliceK);

        #pragma unroll 1
        for (int t = 0; t < tiles; ++t) {
            int future = t+Stages-1;
            if (future < tiles)
                issue_stage<BM,Swizzle,Builtin>(storage+(future%Stages)*StageBytes,
                                               A,B,K,m0,n0,future*SliceK);
            if constexpr (Builtin) {
                int newer = tiles-t-1 < Stages-1 ? tiles-t-1 : Stages-1;
                wait_stage<Ops>(newer);
            } else {
                __syncthreads();
            }
            Fragments f = shared_fragments<BM,Swizzle>(storage+(t%Stages)*StageBytes,wm,wn);
            mma_half(acc,f,0);
            mma_half(acc,f,1);
            // No stage can be overwritten until EVERY wave has finished reading.
            __syncthreads();
        }
    } else {
        // [12] Two register fragment banks. Candidate schedule to be measured.
        #pragma unroll
        for (int s = 0; s < Stages; ++s)
            issue_stage<BM,Swizzle,true>(storage+s*StageBytes,A,B,K,m0,n0,s*SliceK);
        wait_stage<Ops>(Stages-1);
        Fragments current = shared_fragments<BM,Swizzle>(storage,wm,wn);
        #pragma unroll 1
        for (int t = 0; t < tiles; ++t) {
            // Early MMA creates independent work before waiting/loading next.
            mma_half(acc,current,0);
            Fragments next;
            if (t+1 < tiles) {
                int newer = tiles-t-2 < Stages-2 ? tiles-t-2 : Stages-2;
                wait_stage<Ops>(newer);
                next = shared_fragments<BM,Swizzle>(storage+((t+1)%Stages)*StageBytes,wm,wn);
            }
            // Shared reads for next can overlap this independent second half.
            mma_half(acc,current,1);
            __syncthreads();
            if (t+Stages < tiles)
                issue_stage<BM,Swizzle,true>(storage+(t%Stages)*StageBytes,
                                            A,B,K,m0,n0,(t+Stages)*SliceK);
            if (t+1 < tiles) current = next;
        }
    }
    epilogue(acc,C,D,N,m0+64*wm,n0+64*wn,alpha,beta);
}

// [13] Optional row-major B[K,N] -> K-contiguous Bcol[N,K].
// Included in end-to-end timing when --row-b is requested.
__global__ void pack_b_kernel(const __half* row, __half* col, int N, int K) {
    size_t x = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x)*blockDim.x;
    for (; x < size_t(N)*K; x += stride) {
        int n = int(x/K), k = int(x%K);
        col[x] = row[size_t(k)*N+n];
    }
}

static const char* modes[] = {
    "scalar-direct", "scalar-bsm", "builtin-bsm", "xor-wave2x2",
    "stage2", "stage3", "stage4", "stage3-reg-prefetch"
};

// [14] C and D must not alias A/B; D may equal C for a SINGLE invocation.
// Benchmark always keeps C separate, so beta!=0 never accumulates across repeats.
static void launch(int mode, const __half* A, const __half* B, const float* C,
                   float* D, int M, int N, int K, float alpha, float beta) {
    dim3 grid(N/TileN,M/TileM);
    switch (mode) {
        case 0: direct_kernel<128><<<grid,256>>>(A,B,C,D,N,K,alpha,beta); break;
        case 1: staged_kernel<128,1,false,false,false><<<grid,256>>>(A,B,C,D,N,K,alpha,beta); break;
        case 2: staged_kernel<128,1,false,true,false><<<grid,256>>>(A,B,C,D,N,K,alpha,beta); break;
        case 3: staged_kernel<128,1,true,true,false><<<grid,256>>>(A,B,C,D,N,K,alpha,beta); break;
        case 4: staged_kernel<128,2,true,true,false><<<grid,256>>>(A,B,C,D,N,K,alpha,beta); break;
        case 5: staged_kernel<128,3,true,true,false><<<grid,256>>>(A,B,C,D,N,K,alpha,beta); break;
        case 6: staged_kernel<128,4,true,true,false><<<grid,256>>>(A,B,C,D,N,K,alpha,beta); break;
        case 7: staged_kernel<128,3,true,true,true><<<grid,256>>>(A,B,C,D,N,K,alpha,beta); break;
        default: throw std::runtime_error("mode must be 0..7");
    }
    MC_CHECK(mcGetLastError());
}

// [15] Independent SIMT FP64 reference. Full small outputs, stratified large ones.
__global__ void reference_kernel(const __half* A, const __half* B, const float* C,
    double* ref, int M, int N, int K, int count, float alpha, float beta) {
    int t = int(blockIdx.x)*blockDim.x+threadIdx.x;
    if (t >= count) return;
    size_t total = size_t(M)*N;
    size_t index = count == total ? size_t(t) : (size_t(t)*total/count+17)%total;
    int m = int(index/N), n = int(index%N);
    double sum = 0;
    for (int k = 0; k < K; ++k)
        sum += double(__half2float(A[size_t(m)*K+k])) * double(__half2float(B[size_t(n)*K+k]));
    ref[t] = double(alpha)*sum + (beta == 0 ? 0.0 : double(beta)*C[index]);
}

template<class T> struct DeviceArray {
    T* p = nullptr;
    explicit DeviceArray(size_t n) { MC_CHECK(mcMalloc(reinterpret_cast<void**>(&p), n*sizeof(T))); }
    ~DeviceArray() { if (p) mcFree(p); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

template<class F> static double time_ms(F run, int repeats) {
    for (int i=0;i<5;++i) run();
    MC_CHECK(mcDeviceSynchronize());
    mcEvent_t start, stop;
    MC_CHECK(mcEventCreate(&start)); MC_CHECK(mcEventCreate(&stop));
    std::vector<float> times;
    for (int trial=0;trial<7;++trial) {
        MC_CHECK(mcEventRecord(start,0));
        for (int i=0;i<repeats;++i) run();
        MC_CHECK(mcEventRecord(stop,0)); MC_CHECK(mcEventSynchronize(stop));
        float elapsed=0; MC_CHECK(mcEventElapsedTime(&elapsed,start,stop));
        times.push_back(elapsed/repeats);
    }
    MC_CHECK(mcEventDestroy(start)); MC_CHECK(mcEventDestroy(stop));
    std::sort(times.begin(),times.end());
    return times[times.size()/2];
}

// [16] Usage: ./c500_gemm M N K MODE REPEATS ALPHA BETA [--row-b] [--peak=TFLOPS]
// MODE=-1 tests all modes; --pattern=ones|identity|zero are extra correctness cases.
int main(int argc, char** argv) {
    try {
        int M=argc>1?std::stoi(argv[1]):128, N=argc>2?std::stoi(argv[2]):128;
        int K=argc>3?std::stoi(argv[3]):256, mode=argc>4?std::stoi(argv[4]):-1;
        int repeats=argc>5?std::stoi(argv[5]):20;
        float alpha=argc>6?std::stof(argv[6]):1.0f, beta=argc>7?std::stof(argv[7]):1.0f;
        bool row_b=false; double peak=0; std::string pattern="random";
        for (int i=8;i<argc;++i) {
            std::string arg=argv[i];
            if (arg=="--row-b") row_b=true;
            else if (arg.rfind("--peak=",0)==0) peak=std::stod(arg.substr(7));
            else if (arg.rfind("--pattern=",0)==0) pattern=arg.substr(10);
            else throw std::runtime_error("Unknown option: "+arg);
        }
        if (M<=0||N<=0||K<=0||M%TileM||N%TileN||K%TileK||mode < -1||mode>7||repeats<1||
            !std::isfinite(alpha)||!std::isfinite(beta)||!std::isfinite(peak)||peak<0)
            throw std::runtime_error("M,N: positive multiples of 128; K: positive multiple of 256; mode -1..7; repeats>=1; finite scales required");
        if (pattern!="random"&&pattern!="ones"&&pattern!="identity"&&pattern!="zero")
            throw std::runtime_error("Unknown pattern");
        MC_CHECK(mcSetDevice(0));
        mcDeviceProp prop{}; MC_CHECK(mcGetDeviceProperties(&prop,0));
        if (prop.warpSize!=64) throw std::runtime_error("This native-layout kernel requires wave64");
        if (size_t(N/TileN)>size_t(prop.maxGridSize[0]) || size_t(M/TileM)>size_t(prop.maxGridSize[1]))
            throw std::runtime_error("Grid exceeds device limits");
        std::printf("device=%s wave=%d max_threads=%d BSM/block=%zu\n",prop.name,prop.warpSize,
                    prop.maxThreadsPerBlock,size_t(prop.sharedMemPerBlock));

        size_t na=size_t(M)*K, nb=size_t(N)*K, nc=size_t(M)*N;
        if (na>SIZE_MAX/sizeof(__half)||nb>SIZE_MAX/sizeof(__half)||nc>SIZE_MAX/sizeof(float))
            throw std::runtime_error("Allocation size overflow");
        std::vector<__half> ha(na), hb(nb), hr(row_b?nb:0);
        std::vector<float> hc(nc), hd(nc);
        std::mt19937 rng(20261112); std::uniform_real_distribution<float> dist(-0.5f,0.5f);
        for (size_t x=0;x<na;++x) {
            float v=dist(rng);
            if (pattern=="ones") v=1;
            if (pattern=="zero") v=0;
            if (pattern=="identity") v=(x/K==x%K)?1.0f:0.0f;
            ha[x]=__float2half(v);
        }
        for (auto& v:hb) v=__float2half(pattern=="ones"?1.0f:dist(rng));
        for (auto& v:hc) v=dist(rng);
        if (row_b) for(int n=0;n<N;++n) for(int k=0;k<K;++k) hr[size_t(k)*N+n]=hb[size_t(n)*K+k];
        DeviceArray<__half> a(na), b(nb), br(row_b?nb:1);
        DeviceArray<float> c(nc), d(nc);
        MC_CHECK(mcMemcpy(a.p,ha.data(),na*sizeof(__half),mcMemcpyHostToDevice));
        MC_CHECK(mcMemcpy(c.p,hc.data(),nc*sizeof(float),mcMemcpyHostToDevice));
        auto pack=[&] {
            int blocks=int(std::min<size_t>((nb+255)/256,65535));
            pack_b_kernel<<<blocks,256>>>(br.p,b.p,N,K); MC_CHECK(mcGetLastError());
        };
        if (row_b) {
            MC_CHECK(mcMemcpy(br.p,hr.data(),nb*sizeof(__half),mcMemcpyHostToDevice)); pack();
        } else MC_CHECK(mcMemcpy(b.p,hb.data(),nb*sizeof(__half),mcMemcpyHostToDevice));

        int count=int(std::min<size_t>(nc, nc<=262144?nc:4096));
        DeviceArray<double> ref(count); std::vector<double> href(count);
        reference_kernel<<<(count+255)/256,256>>>(a.p,b.p,c.p,ref.p,M,N,K,count,alpha,beta);
        MC_CHECK(mcGetLastError());
        MC_CHECK(mcMemcpy(href.data(),ref.p,count*sizeof(double),mcMemcpyDeviceToHost));
        std::printf("M=%d N=%d K=%d alpha=%g beta=%g check=%s(%d) pattern=%s\n",
            M,N,K,alpha,beta,count==nc?"full":"sampled",count,pattern.c_str());
        std::puts("mode,name,check,max_abs,max_scaled,kernel_ms,TFLOPS,peak_pct,pack_plus_gemm_ms");
        bool failed=false;
        for(int v=(mode<0?0:mode);v<=(mode<0?7:mode);++v) {
            size_t need_shared = v==0?0:
                size_t(v==4?2:v==5||v==7?3:v==6?4:1)*16384;
            if (prop.maxThreadsPerBlock<256 || need_shared>size_t(prop.sharedMemPerBlock)) {
                std::printf("%d,%s,UNSUPPORTED_DEVICE_RESOURCES\n",v,modes[v]);
                if(mode>=0) return 3;
                continue;
            }
            auto run=[&] {launch(v,a.p,b.p,c.p,d.p,M,N,K,alpha,beta);};
            // Poison D so missing stores cannot accidentally pass.
            MC_CHECK(mcMemset(d.p,0xff,nc*sizeof(float))); run();
            MC_CHECK(mcMemcpy(hd.data(),d.p,nc*sizeof(float),mcMemcpyDeviceToHost));
            double max_abs=0,max_scaled=0; int bad=0;
            for(int t=0;t<count;++t) {
                size_t ix=count==nc?size_t(t):(size_t(t)*nc/count+17)%nc;
                double err=std::abs(double(hd[ix])-href[t]);
                // Explicit engineering tolerance, not a universal error bound.
                double tol=2e-3+2e-3*std::abs(href[t]);
                if(!std::isfinite(hd[ix])||err>tol) ++bad;
                max_abs=std::max(max_abs,err); max_scaled=std::max(max_scaled,err/tol);
            }
            if (bad) {
                std::printf("%d,%s,FAIL(%d),%.6g,%.6g,,,,\n",v,modes[v],bad,max_abs,max_scaled);
                failed=true; continue;
            }
            double ms=time_ms(run,repeats);
            double tf=2.0*double(M)*N*K/(ms*1e9);
            double e2e=row_b?time_ms([&]{pack();run();},repeats):ms;
            std::printf("%d,%s,PASS,%.6g,%.6g,%.6f,%.4f,",v,modes[v],max_abs,max_scaled,ms,tf);
            if (peak>0) std::printf("%.3f",100*tf/peak); else std::printf("NA");
            std::printf(",%.6f\n",e2e);
        }
        return failed?1:0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"error: %s\n",e.what()); return 2;
    }
}
