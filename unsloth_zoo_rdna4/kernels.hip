// MXFP4-weight x FP8-activation linears for AMD RDNA4 (gfx12), compiled at run time with hiprtc.
// Header-free on purpose: the serving environment has the HIP runtime and hiprtc, not a compiler or
// HIP headers. The weight layout contract lives in ref.py (repack).
//
// y[M,N] = (fp8 x[M,K] * sx[m]) . (fp8(e2m1 * 2^-d) * 2^(wref[n]-127))^T

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef u32 v4u __attribute__((ext_vector_type(4)));
typedef u32 v2u __attribute__((ext_vector_type(2)));

#define WAVES 8
#define FP8_MAX 448.0f

// Element types by tag: 0 float, 1 half, 2 bfloat16 (16-bit types travel as raw bits).
template <int TAG> struct Elem;
template <> struct Elem<0> {
  typedef float T;
  static __device__ float load(T v) { return v; }
  static __device__ T store(float v) { return v; }
};
template <> struct Elem<1> {
  typedef _Float16 T;
  static __device__ float load(T v) { return (float)v; }
  static __device__ T store(float v) { return (_Float16)v; }
};
template <> struct Elem<2> {
  typedef u16 T;
  static __device__ float load(T v) { return __uint_as_float((u32)v << 16); }
  static __device__ T store(float v) {
    u32 u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    return (u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
};

__device__ inline float row_absmax(float amax) {
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  __shared__ float red[8];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = red[0];
  for (int i = 1; i < 8; i++) amax = fmaxf(amax, red[i]);
  return amax;
}

__device__ inline float clamp8(float v) { return fminf(fmaxf(v, -FP8_MAX), FP8_MAX); }

// Per-token e4m3 quant, row-major output. One block of 256 per row.
template <int TAG>
__device__ void act_quant(const typename Elem<TAG>::T* __restrict__ x, int K, u8* __restrict__ q, float* __restrict__ sx) {
  const typename Elem<TAG>::T* row = x + (size_t)blockIdx.x * K;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += 256) amax = fmaxf(amax, fabsf(Elem<TAG>::load(row[k])));
  float s = fmaxf(row_absmax(amax), 1e-12f) / FP8_MAX;
  if (threadIdx.x == 0) sx[blockIdx.x] = s;
  u8* qr = q + (size_t)blockIdx.x * K;
  for (int k = 2 * threadIdx.x; k < K; k += 512) {
    float a = clamp8(Elem<TAG>::load(row[k]) / s);
    float b = k + 1 < K ? clamp8(Elem<TAG>::load(row[k + 1]) / s) : 0.f;
    u32 p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    qr[k] = p & 0xff;
    if (k + 1 < K) qr[k + 1] = (p >> 8) & 0xff;
  }
}

// Per-token e4m3 quant into WMMA fragment order: [Mp/16, K/64, 32 lanes, 32 bytes], lane l = row l%16,
// bytes 8j..8j+7 = k 64g+16j+8(l/16)+0..7. Rows in [M, Mp) are zero.
template <int TAG>
__device__ void act_quant_tiled(const typename Elem<TAG>::T* __restrict__ x, int M, int K, u8* __restrict__ qt,
                                float* __restrict__ sx) {
  const int m = blockIdx.x, G = K >> 6;
  u8* frag = qt + (size_t)(m >> 4) * G * 1024 + (m & 15) * 32;
  if (m >= M) {
    for (int c = threadIdx.x; c < K / 8; c += 256) {
      int g = c >> 3, j = (c & 7) >> 1, hh = c & 1;
      *(v2u*)(frag + (size_t)g * 1024 + hh * 512 + j * 8) = v2u{0, 0};
    }
    return;
  }
  const typename Elem<TAG>::T* row = x + (size_t)m * K;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += 256) amax = fmaxf(amax, fabsf(Elem<TAG>::load(row[k])));
  float s = fmaxf(row_absmax(amax), 1e-12f) / FP8_MAX;
  if (threadIdx.x == 0) sx[m] = s;
  for (int c = threadIdx.x; c < K / 8; c += 256) {
    u32 w[2];
    for (int p = 0; p < 2; p++) {
      float v[4];
      for (int e = 0; e < 4; e++) v[e] = clamp8(Elem<TAG>::load(row[8 * c + 4 * p + e]) / s);
      u32 lo = __builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], 0, false);
      w[p] = __builtin_amdgcn_cvt_pk_fp8_f32(v[2], v[3], lo, true);
    }
    int g = c >> 3, j = (c & 7) >> 1, hh = c & 1;
    *(v2u*)(frag + (size_t)g * 1024 + hh * 512 + j * 8) = v2u{w[0], w[1]};
  }
}

// 8 repacked fp4 codes -> 8 e4m3 bytes in k order through one LUT row (lo = codes 0-3, hi = 4-7).
__device__ inline v2i dequant8(u32 u, v2u lut) {
  u32 lo = __builtin_amdgcn_perm(lut.y, lut.x, u & 0x07070707u) | ((u << 4) & 0x80808080u);
  u32 hi = __builtin_amdgcn_perm(lut.y, lut.x, (u >> 4) & 0x07070707u) | (u & 0x80808080u);
  return v2i{(int)lo, (int)hi};
}

// Small M with split-K: grid (ceil(T/WAVES), splits, ceil(M/(16*TM))). A wave owns one 16-column tile
// and 16*TM rows. Every lane consumes every WMMA result (a lane-subset consumer lets the compiler run
// WMMA under a partial exec mask).
template <int TM, int OTAG>
__device__ void small_m(const u8* __restrict__ xq, const float* __restrict__ sx, int M, int K,
                        const v4u* __restrict__ wp, const u32* __restrict__ dp, const int* __restrict__ wref,
                        const v2u* __restrict__ lut_g, int N, const typename Elem<OTAG>::T* __restrict__ bias,
                        typename Elem<OTAG>::T* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                        int splits, int gps) {
  typedef Elem<OTAG> E;
  __shared__ v2u lut[16];
  __shared__ int last;
  if (threadIdx.x < 16) lut[threadIdx.x] = lut_g[threadIdx.x];
  __syncthreads();

  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int T = N >> 4, G = K >> 6;
  const int t = blockIdx.x * WAVES + wave;
  const int split = blockIdx.y;
  const int m0 = blockIdx.z * 16 * TM;
  const int g0 = split * gps, g1 = min(G, g0 + gps);
  const int r = lane & 15, h = lane >> 4;

  v8f acc[TM];
  for (int i = 0; i < TM; i++) acc[i] = v8f{};

  if (t < T) {
    // Rows past M clamp to M-1: always-valid loads, results discarded.
    const u8* arow[TM];
    for (int i = 0; i < TM; i++) arow[i] = xq + (size_t)min(m0 + 16 * i + r, M - 1) * K + 8 * h;
    const v4u* wt = wp + (size_t)t * G * 32 + lane;
    const u32* dt = dp + (size_t)t * (K >> 7) * 16 + r;

    v4u wv = wt[(size_t)g0 * 32];
    for (int g = g0; g < g1; g++) {
      v4u wn = wv;
      if (g + 1 < g1) wn = wt[(size_t)(g + 1) * 32];
      u32 dw = dt[(g >> 1) * 16] >> ((g & 1) * 16);
      v2u l0 = lut[dw & 0xf], l1 = lut[(dw >> 8) & 0xf];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        v2i b = dequant8(wv[j], j < 2 ? l0 : l1);
        const int k = 64 * g + 16 * j;
#pragma unroll
        for (int i = 0; i < TM; i++) {
          v2i a = *(const v2i*)(arow[i] + k);
          acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc[i]);
        }
      }
      wv = wn;
    }
  }

  const int n = t * 16 + r;
  const float wscale = t < T ? exp2f((float)(wref[n] - 127)) : 0.f;
  const float bv = (t < T && bias) ? E::load(bias[n]) : 0.f;
  if (splits == 1) {
    if (t < T)
      for (int i = 0; i < TM; i++)
        for (int e = 0; e < 8; e++) {
          int m = m0 + 16 * i + 8 * h + e;
          if (m < M) out[(size_t)m * N + n] = E::store(acc[i][e] * sx[m] * wscale + bv);
        }
    return;
  }

  // Split-K: write fp32 partials; the last block of this (x, z) column reduces them in split order.
  const size_t plane = (size_t)M * N;
  if (t < T)
    for (int i = 0; i < TM; i++)
      for (int e = 0; e < 8; e++) {
        int m = m0 + 16 * i + 8 * h + e;
        if (m < M) ws[split * plane + (size_t)m * N + n] = acc[i][e];
      }
  __threadfence();
  __syncthreads();
  int* ctr = counters + blockIdx.z * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) last = atomicAdd(ctr, 1) == splits - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (t < T)
    for (int i = 0; i < TM; i++)
      for (int e = 0; e < 8; e++) {
        int m = m0 + 16 * i + 8 * h + e;
        if (m < M) {
          float v = 0.f;
          for (int s = 0; s < splits; s++) v += __builtin_nontemporal_load(ws + s * plane + (size_t)m * N + n);
          out[(size_t)m * N + n] = E::store(v * sx[m] * wscale + bv);
        }
      }
  if (threadIdx.x == 0) *ctr = 0;
}

// Large M: block 128(m) x 256(n), 8 waves as 2(m) x 4(n), wave tile 64x64 (16 accumulators keep fp8
// WMMA near its issue rate). Both operands load in fragment order; L1 serves the cross-wave reuse.
// m_fast walks m fastest so consecutive blocks share a weight tile (weights larger than the MALL).
template <int OTAG>
__device__ void large_m(const u8* __restrict__ qt, const float* __restrict__ sx, int M, int K,
                        const v4u* __restrict__ wp, const u32* __restrict__ dp, const int* __restrict__ wref,
                        const v2u* __restrict__ lut_g, int N, const typename Elem<OTAG>::T* __restrict__ bias,
                        typename Elem<OTAG>::T* __restrict__ out, int m_fast) {
  typedef Elem<OTAG> E;
  __shared__ v2u lut[16];
  if (threadIdx.x < 16) lut[threadIdx.x] = lut_g[threadIdx.x];
  __syncthreads();

  const int bm = m_fast ? blockIdx.x : blockIdx.y, bn = m_fast ? blockIdx.y : blockIdx.x;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int wm = wave >> 2, wn = wave & 3;
  const int T = N >> 4, G = K >> 6;
  const int r = lane & 15, h = lane >> 4;
  const int f0 = bm * 8 + wm * 4, t0 = bn * 16 + wn * 4;
  const int Fp = (M + 15) >> 4;

  const v4u* wt[4];
  const u32* dt[4];
  const v4u* at[4];
  for (int i = 0; i < 4; i++) {
    int t = min(t0 + i, T - 1);
    wt[i] = wp + (size_t)t * G * 32 + lane;
    dt[i] = dp + (size_t)t * (K >> 7) * 16 + r;
    at[i] = (const v4u*)(qt + (size_t)min(f0 + i, Fp - 1) * G * 1024 + lane * 32);
  }

  v8f acc[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) acc[i][j] = v8f{};

  for (int g = 0; g < G; g++) {
    v4u wv[4], a0[4], a1[4];
    u32 dw[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      wv[i] = wt[i][(size_t)g * 32];
      dw[i] = dt[i][(g >> 1) * 16] >> ((g & 1) * 16);
      a0[i] = at[i][(size_t)g * 64];
      a1[i] = at[i][(size_t)g * 64 + 1];
    }
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      v2i b[4];
#pragma unroll
      for (int n = 0; n < 4; n++) b[n] = dequant8(wv[n][ks], lut[ks < 2 ? (dw[n] & 0xf) : ((dw[n] >> 8) & 0xf)]);
#pragma unroll
      for (int m = 0; m < 4; m++) {
        v4u av = ks < 2 ? a0[m] : a1[m];
        v2i a = v2i{(int)av[(ks & 1) * 2], (int)av[(ks & 1) * 2 + 1]};
#pragma unroll
        for (int n = 0; n < 4; n++) acc[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b[n], acc[m][n]);
      }
    }
  }

  for (int n = 0; n < 4; n++) {
    int t = t0 + n;
    if (t >= T) continue;
    int col = t * 16 + r;
    float wscale = exp2f((float)(wref[col] - 127));
    float bv = bias ? E::load(bias[col]) : 0.f;
    for (int m = 0; m < 4; m++)
      for (int e = 0; e < 8; e++) {
        int row = (f0 + m) * 16 + 8 * h + e;
        if (row < M) out[(size_t)row * N + col] = E::store(acc[m][n][e] * sx[row] * wscale + bv);
      }
  }
}

#define LB __launch_bounds__(256)
#define ACT(TAG)                                                                                          \
  extern "C" __global__ LB void act_quant_##TAG(const void* x, int K, u8* q, float* sx) {                 \
    act_quant<TAG>((const Elem<TAG>::T*)x, K, q, sx);                                                     \
  }                                                                                                       \
  extern "C" __global__ LB void act_quant_tiled_##TAG(const void* x, int M, int K, u8* qt, float* sx) {   \
    act_quant_tiled<TAG>((const Elem<TAG>::T*)x, M, K, qt, sx);                                           \
  }                                                                                                       \
  extern "C" __global__ LB void large_m_##TAG(const u8* qt, const float* sx, int M, int K, const v4u* wp,  \
                                              const u32* dp, const int* wref, const v2u* lut, int N,      \
                                              const void* bias, void* out, int m_fast) {                  \
    large_m<TAG>(qt, sx, M, K, wp, dp, wref, lut, N, (const Elem<TAG>::T*)bias, (Elem<TAG>::T*)out, m_fast); \
  }
#define SMALL(TM, TAG)                                                                                    \
  extern "C" __global__ LB void small_m_##TM##_##TAG(const u8* xq, const float* sx, int M, int K,         \
                                                     const v4u* wp, const u32* dp, const int* wref,       \
                                                     const v2u* lut, int N, const void* bias, void* out,  \
                                                     float* ws, int* counters, int splits, int gps) {     \
    small_m<TM, TAG>(xq, sx, M, K, wp, dp, wref, lut, N, (const Elem<TAG>::T*)bias, (Elem<TAG>::T*)out,   \
                     ws, counters, splits, gps);                                                          \
  }

ACT(0) ACT(1) ACT(2)
SMALL(1, 0) SMALL(1, 1) SMALL(1, 2)
SMALL(2, 0) SMALL(2, 1) SMALL(2, 2)
SMALL(3, 0) SMALL(3, 1) SMALL(3, 2)
SMALL(4, 0) SMALL(4, 1) SMALL(4, 2)
