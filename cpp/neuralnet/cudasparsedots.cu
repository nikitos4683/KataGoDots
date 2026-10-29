// CUDA kernels of the sparse Dots models (model version 18), see cudasparsedots.h.

#include "../neuralnet/cudasparsedots.h"

#include <climits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../neuralnet/activations.h"

static int numBlocksFor(long long numThreads, int blockSize) {
  return (int)((numThreads + blockSize - 1) / blockSize);
}

__device__ __forceinline__ float toFloat(float x) { return x; }
__device__ __forceinline__ float toFloat(half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T fromFloat(float x);
template <>
__device__ __forceinline__ float fromFloat<float>(float x) { return x; }
template <>
__device__ __forceinline__ half fromFloat<half>(float x) { return __float2half(x); }

__device__ __forceinline__ float activate(float x, int activation) {
  switch(activation) {
    case ACTIVATION_RELU: return fmaxf(x, 0.0f);
    case ACTIVATION_SILU: return x / (1.0f + __expf(-x));
    case ACTIVATION_MISH: return x * tanhf(x < 20.0f ? log1pf(__expf(x)) : x);
    default: return x;
  }
}

// The elementwise kernels give each thread VEC consecutive elements of a row (one 16-byte load for half), so the
// channels, the row strides and the offsets of the heads must be multiples of VEC (checked on the host). The
// indexes are 32-bit: a 64-bit division or remainder per element cost more than the arithmetic of these kernels.
constexpr int VEC = 8;

__device__ __forceinline__ void loadVec(const half* p, float* v) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const half2* h = reinterpret_cast<const half2*>(&raw);
  #pragma unroll
  for(int i = 0; i < VEC / 2; i++) {
    const float2 f = __half22float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}
__device__ __forceinline__ void loadVec(const float* p, float* v) {
  const float4 a = reinterpret_cast<const float4*>(p)[0];
  const float4 b = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
  v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void storeVec(half* p, const float* v) {
  uint4 raw;
  half2* h = reinterpret_cast<half2*>(&raw);
  #pragma unroll
  for(int i = 0; i < VEC / 2; i++)
    h[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = raw;
}
__device__ __forceinline__ void storeVec(float* p, const float* v) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}

static void checkVec(long long rows, int channels, const char* name) {
  if(channels % VEC != 0)
    throw std::runtime_error(std::string(name) + ": channels must be a multiple of 8");
  if(rows * channels / VEC > INT_MAX)
    throw std::runtime_error(std::string(name) + ": too many rows");
}

//------------------------------------------------------------------------------------------------------------------

// A vector per thread, the last thread converts the elements after the last full vector one by one
__global__ void sparseDotsCopyFromHalfKernel(const half* in, float* out, int n) {
  const int begin = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  if(begin + VEC <= n) {
    float x[VEC];
    loadVec(in + begin, x);
    storeVec(out + begin, x);
  }
  else {
    for(int i = begin; i < n; i++)
      out[i] = __half2float(in[i]);
  }
}

void sparseDotsCopyFromHalf(const half* in, float* out, long long n, cudaStream_t stream) {
  if(n + VEC > INT_MAX)
    throw std::runtime_error("sparseDotsCopyFromHalf: too many elements");
  if(n > 0)
    sparseDotsCopyFromHalfKernel<<<numBlocksFor((n + VEC - 1) / VEC, 256), 256, 0, stream>>>(in, out, (int)n);
}

//------------------------------------------------------------------------------------------------------------------

// A thread computes VEC channels of a row: the features of the row are loaded once for all of them
template <typename T>
__global__ void sparseDotsEmbedTokensKernel(
  T* out, const float* pointFeat, const float* weight, const float* bias, int numFeatures, const float* globalToken,
  const float* globalFeat, const float* globalWeight, int numGlobalFeatures, const int* rowSeg, const int* segStart,
  int numVecs, int rowVecs, int channels
) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= numVecs)
    return;
  const int r = v / rowVecs;
  const int c = (v - r * rowVecs) * VEC;
  const int n = rowSeg[r];
  float x[VEC];
  if(r == segStart[n]) {
    loadVec(globalToken + c, x);
  }
  else {
    // Every position has one GLOBAL row before its points
    const float* f = pointFeat + (size_t)(r - n - 1) * numFeatures;
    loadVec(bias + c, x);
    for(int k = 0; k < numFeatures; k++) {
      const float fk = f[k];
      float w[VEC];
      loadVec(weight + (size_t)k * channels + c, w);
      #pragma unroll
      for(int j = 0; j < VEC; j++)
        x[j] += fk * w[j];
    }
  }
  const float* gf = globalFeat + (size_t)n * numGlobalFeatures;
  for(int k = 0; k < numGlobalFeatures; k++) {
    const float fk = gf[k];
    float w[VEC];
    loadVec(globalWeight + (size_t)k * channels + c, w);
    #pragma unroll
    for(int j = 0; j < VEC; j++)
      x[j] += fk * w[j];
  }
  storeVec(out + (size_t)v * VEC, x);
}

template <typename T>
void sparseDotsEmbedTokens(
  T* out, const float* pointFeat, const float* weight, const float* bias, int numFeatures, const float* globalToken,
  const float* globalFeat, const float* globalWeight, int numGlobalFeatures, const int* rowSeg, const int* segStart,
  int rows, int channels, cudaStream_t stream
) {
  checkVec(rows, channels, "sparseDotsEmbedTokens");
  const int numVecs = (int)((long long)rows * channels / VEC);
  if(numVecs > 0)
    sparseDotsEmbedTokensKernel<T><<<numBlocksFor(numVecs, 256), 256, 0, stream>>>(
      out, pointFeat, weight, bias, numFeatures, globalToken, globalFeat, globalWeight, numGlobalFeatures, rowSeg,
      segStart, numVecs, channels / VEC, channels);
}

template <typename T>
__global__ void sparseDotsEmbedQueriesKernel(
  T* out, const float* candFeat, const float* candWeight, const float* candBias, int numCandFeatures,
  const float* groundFeat, const float* groundWeight, const float* groundBias, const float* poolProj, const int* rowSeg,
  const int* rowSrc, int numVecs, int rowVecs, int channels
) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= numVecs)
    return;
  const int r = v / rowVecs;
  const int c = (v - r * rowVecs) * VEC;
  const int src = rowSrc[r];
  float x[VEC], w[VEC];
  if(src >= 0) {
    const float* f = candFeat + (size_t)src * numCandFeatures;
    loadVec(candBias + c, x);
    for(int k = 0; k < numCandFeatures; k++) {
      const float fk = f[k];
      loadVec(candWeight + (size_t)k * channels + c, w);
      #pragma unroll
      for(int j = 0; j < VEC; j++)
        x[j] += fk * w[j];
    }
  }
  else {
    const float fg = groundFeat[-src - 1];
    loadVec(groundBias + c, x);
    loadVec(groundWeight + c, w);
    #pragma unroll
    for(int j = 0; j < VEC; j++)
      x[j] += fg * w[j];
  }
  loadVec(poolProj + (size_t)rowSeg[r] * channels + c, w);
  #pragma unroll
  for(int j = 0; j < VEC; j++)
    x[j] += w[j];
  storeVec(out + (size_t)v * VEC, x);
}

template <typename T>
void sparseDotsEmbedQueries(
  T* out, const float* candFeat, const float* candWeight, const float* candBias, int numCandFeatures,
  const float* groundFeat, const float* groundWeight, const float* groundBias, const float* poolProj, const int* rowSeg,
  const int* rowSrc, int rows, int channels, cudaStream_t stream
) {
  checkVec(rows, channels, "sparseDotsEmbedQueries");
  const int numVecs = (int)((long long)rows * channels / VEC);
  if(numVecs > 0)
    sparseDotsEmbedQueriesKernel<T><<<numBlocksFor(numVecs, 256), 256, 0, stream>>>(
      out, candFeat, candWeight, candBias, numCandFeatures, groundFeat, groundWeight, groundBias, poolProj, rowSeg,
      rowSrc, numVecs, channels / VEC, channels);
}

//------------------------------------------------------------------------------------------------------------------

template <typename T>
__global__ void sparseDotsScaleBiasActKernel(
  const T* in, T* out, const float* scale, const float* bias, int numVecs, int rowVecs, int activation
) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= numVecs)
    return;
  const int c = (v % rowVecs) * VEC;
  float x[VEC];
  loadVec(in + (size_t)v * VEC, x);
  #pragma unroll
  for(int j = 0; j < VEC; j++)
    x[j] = activate(x[j] * scale[c + j] + bias[c + j], activation);
  storeVec(out + (size_t)v * VEC, x);
}

template <typename T>
void sparseDotsScaleBiasAct(
  const T* in, T* out, const float* scale, const float* bias, long long rows, int channels, int activation,
  cudaStream_t stream
) {
  checkVec(rows, channels, "sparseDotsScaleBiasAct");
  const int numVecs = (int)(rows * channels / VEC);
  if(numVecs > 0)
    sparseDotsScaleBiasActKernel<T><<<numBlocksFor(numVecs, 256), 256, 0, stream>>>(
      in, out, scale, bias, numVecs, channels / VEC, activation);
}

template <typename T>
__global__ void sparseDotsBiasActKernel(T* buf, const float* bias, long long rows, int channels, int activation) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= rows * channels)
    return;
  float v = toFloat(buf[i]);
  if(bias != nullptr)
    v += bias[i % channels];
  buf[i] = fromFloat<T>(activate(v, activation));
}

template <typename T>
void sparseDotsBiasAct(T* buf, const float* bias, long long rows, int channels, int activation, cudaStream_t stream) {
  long long n = rows * channels;
  if(n > 0 && (bias != nullptr || activation != ACTIVATION_IDENTITY))
    sparseDotsBiasActKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(buf, bias, rows, channels, activation);
}

// One warp per row
template <typename T>
__global__ void sparseDotsRMSNormKernel(
  const T* in, T* out, const float* weight, int rows, int channels, float epsilon
) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if(row >= rows)
    return;
  const T* x = in + (size_t)row * channels;
  float sumSq = 0.0f;
  float v[VEC];
  for(int c = lane * VEC; c < channels; c += 32 * VEC) {
    loadVec(x + c, v);
    #pragma unroll
    for(int j = 0; j < VEC; j++)
      sumSq += v[j] * v[j];
  }
  for(int offset = 16; offset > 0; offset /= 2)
    sumSq += __shfl_xor_sync(0xffffffff, sumSq, offset);
  const float inv = rsqrtf(sumSq / (float)channels + epsilon);
  T* o = out + (size_t)row * channels;
  for(int c = lane * VEC; c < channels; c += 32 * VEC) {
    loadVec(x + c, v);
    #pragma unroll
    for(int j = 0; j < VEC; j++)
      v[j] = v[j] * inv * weight[c + j];
    storeVec(o + c, v);
  }
}

template <typename T>
void sparseDotsRMSNorm(const T* in, T* out, const float* weight, long long rows, int channels, float epsilon, cudaStream_t stream) {
  checkVec(rows, channels, "sparseDotsRMSNorm");
  if(rows * 32 > INT_MAX)
    throw std::runtime_error("sparseDotsRMSNorm: too many rows");
  if(rows > 0)
    sparseDotsRMSNormKernel<T><<<numBlocksFor(rows * 32, 128), 128, 0, stream>>>(in, out, weight, (int)rows, channels, epsilon);
}

template <typename T>
__global__ void sparseDotsSwiGLUKernel(T* ab, int numVecs, int rowVecs, int channels) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= numVecs)
    return;
  const int r = v / rowVecs;
  const int c = (v - r * rowVecs) * VEC;
  T* a = ab + (size_t)r * 2 * channels + c;
  float x[VEC], g[VEC];
  loadVec(a, x);
  loadVec(a + channels, g);
  #pragma unroll
  for(int j = 0; j < VEC; j++)
    x[j] = x[j] / (1.0f + __expf(-x[j])) * g[j];
  storeVec(a, x);
}

template <typename T>
void sparseDotsSwiGLU(T* ab, long long rows, int channels, cudaStream_t stream) {
  checkVec(rows, channels, "sparseDotsSwiGLU");
  const int numVecs = (int)(rows * channels / VEC);
  if(numVecs > 0)
    sparseDotsSwiGLUKernel<T><<<numBlocksFor(numVecs, 256), 256, 0, stream>>>(ab, numVecs, channels / VEC, channels);
}

//------------------------------------------------------------------------------------------------------------------

constexpr int SPATIAL_RMSNORM_BLOCKS_PER_SEG = 8;

template <typename T>
__global__ void sparseDotsSpatialSumSqKernel(const T* in, const int* segStart, int channels, float* sumSqBuf) {
  __shared__ float shared[256];
  const int n = blockIdx.y;
  const int begin = segStart[n] * (channels / VEC);
  const int end = segStart[n + 1] * (channels / VEC);
  float acc = 0.0f;
  float x[VEC];
  for(int v = begin + blockIdx.x * blockDim.x + threadIdx.x; v < end; v += blockDim.x * gridDim.x) {
    loadVec(in + (size_t)v * VEC, x);
    #pragma unroll
    for(int j = 0; j < VEC; j++)
      acc += x[j] * x[j];
  }
  shared[threadIdx.x] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s /= 2) {
    if(threadIdx.x < s)
      shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  if(threadIdx.x == 0)
    atomicAdd(sumSqBuf + n, shared[0]);
}

template <typename T>
__global__ void sparseDotsSpatialApplyKernel(
  const T* in, T* out, float* outFloat, const float* gamma, const float* beta, const int* rowSeg, const int* segStart,
  int numVecs, int rowVecs, int channels, float epsilon, int activation, const float* sumSqBuf
) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= numVecs)
    return;
  const int r = v / rowVecs;
  const int c = (v - r * rowVecs) * VEC;
  const int n = rowSeg[r];
  const float numTokens = (float)(segStart[n + 1] - segStart[n]);
  const float inv = rsqrtf(sumSqBuf[n] / (numTokens * (float)channels) + epsilon);
  float x[VEC];
  loadVec(in + (size_t)v * VEC, x);
  #pragma unroll
  for(int j = 0; j < VEC; j++)
    x[j] = activate(x[j] * inv * gamma[c + j] + beta[c + j], activation);
  storeVec(out + (size_t)v * VEC, x);
  if(outFloat != nullptr) {
    // The FP32 copy has the values rounded like out, as a separate conversion of out would
    if constexpr(std::is_same<T, half>::value) {
      #pragma unroll
      for(int j = 0; j < VEC; j++)
        x[j] = __half2float(__float2half(x[j]));
    }
    storeVec(outFloat + (size_t)v * VEC, x);
  }
}

template <typename T>
void sparseDotsSpatialRMSNorm(
  const T* in, T* out, float* outFloat, const float* gamma, const float* beta, const int* rowSeg, const int* segStart,
  int numSegs, int rows, int channels, float epsilon, int activation, float* sumSqBuf, cudaStream_t stream
) {
  if(numSegs > 65535)
    throw std::runtime_error("sparseDotsSpatialRMSNorm: too many positions");
  checkVec(rows, channels, "sparseDotsSpatialRMSNorm");
  if(rows <= 0)
    return;
  cudaMemsetAsync(sumSqBuf, 0, sizeof(float) * numSegs, stream);
  dim3 grid(SPATIAL_RMSNORM_BLOCKS_PER_SEG, numSegs);
  sparseDotsSpatialSumSqKernel<T><<<grid, 256, 0, stream>>>(in, segStart, channels, sumSqBuf);
  const int numVecs = (int)((long long)rows * channels / VEC);
  sparseDotsSpatialApplyKernel<T><<<numBlocksFor(numVecs, 256), 256, 0, stream>>>(
    in, out, outFloat, gamma, beta, rowSeg, segStart, numVecs, channels / VEC, channels, epsilon, activation, sumSqBuf);
}

//------------------------------------------------------------------------------------------------------------------
// RoPE: a thread rotates the VEC / 2 consecutive pairs of a head in one vector

static void checkRope(int rows, int numHeads, int headDim, int rowStride, const char* name) {
  if(headDim % VEC != 0 || rowStride % VEC != 0)
    throw std::runtime_error(std::string(name) + ": head dim and row stride must be multiples of 8");
  if((long long)rows * numHeads * (headDim / VEC) > INT_MAX)
    throw std::runtime_error(std::string(name) + ": too many rows");
}

template <typename T>
__device__ __forceinline__ void sparseDotsRotateVec(T* p, const float* c, const float* s) {
  float x[VEC];
  loadVec(p, x);
  #pragma unroll
  for(int j = 0; j < VEC / 2; j++) {
    const float x0 = x[2 * j];
    const float x1 = x[2 * j + 1];
    x[2 * j] = x0 * c[j] - x1 * s[j];
    x[2 * j + 1] = x0 * s[j] + x1 * c[j];
  }
  storeVec(p, x);
}

// The angles of the pairs [firstPair, firstPair + VEC / 2) of kv head kvh at the row position, zero if none
__device__ __forceinline__ void sparseDotsRopeAngles(
  const float* rowPos, const float* freqs, int kvh, int numPairs, int firstPair, float* c, float* s
) {
  const float* f = freqs + 2 * (kvh * numPairs + firstPair);
  #pragma unroll
  for(int j = 0; j < VEC / 2; j++) {
    if(rowPos[2] != 0.0f) {
      sincosf(rowPos[0] * f[2 * j] + rowPos[1] * f[2 * j + 1], &s[j], &c[j]);
    }
    else {
      s[j] = 0.0f;
      c[j] = 0.0f;
    }
  }
}

template <typename T>
__global__ void sparseDotsApplyRopeKernel(
  T* x, const float* freqs, const float* pos, int numVecs, int numHeads, int numKVHeads, int headDim, int rowStride
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= numVecs)
    return;
  const int headVecs = headDim / VEC;
  const int pv = i % headVecs;
  const int rh = i / headVecs;
  const int h = rh % numHeads;
  const int r = rh / numHeads;
  float c[VEC / 2], s[VEC / 2];
  // Rows without a position become zero: c = s = 0
  sparseDotsRopeAngles(pos + r * 3, freqs, h * numKVHeads / numHeads, headDim / 2, pv * VEC / 2, c, s);
  sparseDotsRotateVec(x + (size_t)r * rowStride + h * headDim + pv * VEC, c, s);
}

template <typename T>
void sparseDotsApplyRope(
  T* x, const float* freqs, const float* pos, int rows, int numHeads, int numKVHeads, int headDim, int rowStride,
  cudaStream_t stream
) {
  if(rows <= 0)
    return;
  checkRope(rows, numHeads, headDim, rowStride, "sparseDotsApplyRope");
  const int numVecs = rows * numHeads * (headDim / VEC);
  sparseDotsApplyRopeKernel<T><<<numBlocksFor(numVecs, 256), 256, 0, stream>>>(
    x, freqs, pos, numVecs, numHeads, numKVHeads, headDim, rowStride);
}

template <typename T>
__global__ void sparseDotsApplyRopeQKKernel(
  T* q, T* k, const float* freqs, const float* pos, int numVecs, int numHeads, int numKVHeads, int headDim, int qStride,
  int kStride
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= numVecs)
    return;
  const int headVecs = headDim / VEC;
  const int pv = i % headVecs;
  const int rh = i / headVecs;
  const int kvh = rh % numKVHeads;
  const int r = rh / numKVHeads;
  float c[VEC / 2], s[VEC / 2];
  sparseDotsRopeAngles(pos + r * 3, freqs, kvh, headDim / 2, pv * VEC / 2, c, s);
  sparseDotsRotateVec(k + (size_t)r * kStride + kvh * headDim + pv * VEC, c, s);
  const int rep = numHeads / numKVHeads;
  for(int j = 0; j < rep; j++)
    sparseDotsRotateVec(q + (size_t)r * qStride + (kvh * rep + j) * headDim + pv * VEC, c, s);
}

template <typename T>
void sparseDotsApplyRopeQK(
  T* q, T* k, const float* freqs, const float* pos, int rows, int numHeads, int numKVHeads, int headDim, int qStride,
  int kStride, cudaStream_t stream
) {
  if(rows <= 0)
    return;
  if(numHeads % numKVHeads != 0)
    throw std::runtime_error("sparseDotsApplyRopeQK: numHeads must divide by numKVHeads");
  checkRope(rows, numHeads, headDim, qStride, "sparseDotsApplyRopeQK");
  checkRope(rows, numKVHeads, headDim, kStride, "sparseDotsApplyRopeQK");
  const int numVecs = rows * numKVHeads * (headDim / VEC);
  sparseDotsApplyRopeQKKernel<T><<<numBlocksFor(numVecs, 256), 256, 0, stream>>>(
    q, k, freqs, pos, numVecs, numHeads, numKVHeads, headDim, qStride, kStride);
}

//------------------------------------------------------------------------------------------------------------------
// Attention with online softmax, tiled like flashAttentionTiledImpl in cudaandrocmhelpers.inc but over the packed
// positions. One thread per query, BLOCK_KV keys in shared memory. Used when the tensor-core kernel isn't.

template <typename T, int qHeadDim, int vHeadDim, int BLOCK_Q, int BLOCK_KV>
__global__ void sparseDotsAttentionKernel(
  const T* Q, const T* K, const T* V, T* output, const int* qStart, const int* kvStart, int numHeads, int numKVHeads,
  int qStride, int kvStride, float scale, const int* chainCodes, const float* chainBias
) {
  const int tid = threadIdx.x;
  const int bh = blockIdx.y;
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h * numKVHeads / numHeads;
  const int qBase = qStart[n];
  const int qLen = qStart[n + 1] - qBase;
  const int kvBase = kvStart[n];
  const int kvLen = kvStart[n + 1] - kvBase;
  const int qBlockStart = blockIdx.x * BLOCK_Q;
  if(qBlockStart >= qLen)
    return;
  const int qPos = qBlockStart + tid;
  const int oTotalDim = numHeads * vHeadDim;

  __shared__ float kTile[BLOCK_KV * qHeadDim];
  __shared__ float vTile[BLOCK_KV * vHeadDim];
  __shared__ int codeTile[BLOCK_KV];

  float qReg[qHeadDim];
  float acc[vHeadDim];
  float runningMax = -1e30f;
  float runningSum = 0.0f;
  const bool active = qPos < qLen;
  if(active) {
    const T* qPtr = Q + (long long)(qBase + qPos) * qStride + h * qHeadDim;
    #pragma unroll
    for(int d = 0; d < qHeadDim; d++)
      qReg[d] = toFloat(qPtr[d]) * scale;
  }
  #pragma unroll
  for(int d = 0; d < vHeadDim; d++)
    acc[d] = 0.0f;
  int rowCode = -1;
  float bias0 = 0.0f, bias1 = 0.0f, bias2 = 0.0f;
  if(chainCodes != nullptr) {
    if(active)
      rowCode = chainCodes[qBase + qPos];
    bias0 = chainBias[h * 3 + 0];
    bias1 = chainBias[h * 3 + 1];
    bias2 = chainBias[h * 3 + 2];
  }

  for(int kvTileStart = 0; kvTileStart < kvLen; kvTileStart += BLOCK_KV) {
    for(int t = tid; t < BLOCK_KV * qHeadDim; t += BLOCK_Q) {
      int p = kvTileStart + t / qHeadDim;
      kTile[t] = p < kvLen ? toFloat(K[(long long)(kvBase + p) * kvStride + kvh * qHeadDim + t % qHeadDim]) : 0.0f;
    }
    for(int t = tid; t < BLOCK_KV * vHeadDim; t += BLOCK_Q) {
      int p = kvTileStart + t / vHeadDim;
      vTile[t] = p < kvLen ? toFloat(V[(long long)(kvBase + p) * kvStride + kvh * vHeadDim + t % vHeadDim]) : 0.0f;
    }
    if(chainCodes != nullptr) {
      for(int t = tid; t < BLOCK_KV; t += BLOCK_Q) {
        int p = kvTileStart + t;
        codeTile[t] = p < kvLen ? chainCodes[kvBase + p] : -1;
      }
    }
    __syncthreads();
    if(active) {
      int kvEnd = min(BLOCK_KV, kvLen - kvTileStart);
      for(int tk = 0; tk < kvEnd; tk++) {
        float dot = 0.0f;
        #pragma unroll
        for(int d = 0; d < qHeadDim; d++)
          dot += qReg[d] * kTile[tk * qHeadDim + d];
        if(rowCode >= 0) {
          const int keyCode = codeTile[tk];
          if(keyCode >= 0)
            dot += keyCode == rowCode ? bias0 : (((keyCode ^ rowCode) & 1) ? bias2 : bias1);
        }
        float newMax = fmaxf(runningMax, dot);
        float expOld = __expf(runningMax - newMax);
        float expCur = __expf(dot - newMax);
        #pragma unroll
        for(int d = 0; d < vHeadDim; d++)
          acc[d] = acc[d] * expOld + expCur * vTile[tk * vHeadDim + d];
        runningSum = runningSum * expOld + expCur;
        runningMax = newMax;
      }
    }
    __syncthreads();
  }

  if(active) {
    T* outRow = output + (long long)(qBase + qPos) * oTotalDim + h * vHeadDim;
    float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
    #pragma unroll
    for(int d = 0; d < vHeadDim; d++)
      outRow[d] = fromFloat<T>(acc[d] * invSum);
  }
}

template <typename T>
void sparseDotsAttention(
  const T* Q, const T* K, const T* V, T* output, const int* qStart, const int* kvStart, int numSegs, int maxQLen,
  int numHeads, int numKVHeads, int qHeadDim, int vHeadDim, int qStride, int kvStride, const int* chainCodes,
  const float* chainBias, cudaStream_t stream
) {
  if((long long)numSegs * numHeads > 65535)
    throw std::runtime_error("sparseDotsAttention: too many positions * heads");
  if(maxQLen <= 0 || numSegs <= 0)
    return;
  float scale = 1.0f / sqrtf((float)qHeadDim);
  constexpr int BLOCK_Q = 64;
  constexpr int BLOCK_KV = 32;
  dim3 grid((maxQLen + BLOCK_Q - 1) / BLOCK_Q, numSegs * numHeads);
  if(qHeadDim == 32 && vHeadDim == 32)
    sparseDotsAttentionKernel<T, 32, 32, BLOCK_Q, BLOCK_KV><<<grid, BLOCK_Q, 0, stream>>>(
      Q, K, V, output, qStart, kvStart, numHeads, numKVHeads, qStride, kvStride, scale, chainCodes, chainBias);
  else if(qHeadDim == 64 && vHeadDim == 64)
    sparseDotsAttentionKernel<T, 64, 64, BLOCK_Q, BLOCK_KV><<<grid, BLOCK_Q, 0, stream>>>(
      Q, K, V, output, qStart, kvStart, numHeads, numKVHeads, qStride, kvStride, scale, chainCodes, chainBias);
  else if(qHeadDim == 16 && vHeadDim == 16)
    sparseDotsAttentionKernel<T, 16, 16, BLOCK_Q, BLOCK_KV><<<grid, BLOCK_Q, 0, stream>>>(
      Q, K, V, output, qStart, kvStart, numHeads, numKVHeads, qStride, kvStride, scale, chainCodes, chainBias);
  else
    throw std::runtime_error("sparseDotsAttention: unsupported head dims");
}

//------------------------------------------------------------------------------------------------------------------

// A block per position and 32 channels, the rows split over POOL_SLICES rows of threads
constexpr int POOL_SLICES = 8;

__global__ void sparseDotsPoolKernel(
  const float* x, const int* segStart, const float* boardOffset, float* out, int channels, bool valueHeadMode
) {
  __shared__ float partialSum[POOL_SLICES][32];
  __shared__ float partialMax[POOL_SLICES][32];
  const int n = blockIdx.y;
  const int c = blockIdx.x * 32 + threadIdx.x;
  const int begin = segStart[n];
  const int end = segStart[n + 1];
  float sum = 0.0f;
  float maxV = -1e30f;
  if(c < channels) {
    for(int r = begin + threadIdx.y; r < end; r += POOL_SLICES) {
      const float v = x[(size_t)r * channels + c];
      sum += v;
      maxV = fmaxf(maxV, v);
    }
  }
  partialSum[threadIdx.y][threadIdx.x] = sum;
  partialMax[threadIdx.y][threadIdx.x] = maxV;
  __syncthreads();
  if(threadIdx.y != 0 || c >= channels)
    return;
  for(int s = 1; s < POOL_SLICES; s++) {
    sum += partialSum[s][threadIdx.x];
    maxV = fmaxf(maxV, partialMax[s][threadIdx.x]);
  }
  const float mean = sum / (float)(end - begin);
  const float off = boardOffset[n];
  float* o = out + (size_t)n * 3 * channels;
  o[c] = mean;
  o[channels + c] = mean * (off / 10.0f);
  o[2 * channels + c] = valueHeadMode ? mean * (off * off / 100.0f - 0.1f) : maxV;
}

void sparseDotsPool(
  const float* x, const int* segStart, const float* boardOffset, float* out, int numSegs, int channels, bool valueHeadMode,
  cudaStream_t stream
) {
  if(numSegs > 65535)
    throw std::runtime_error("sparseDotsPool: too many positions");
  if(numSegs <= 0)
    return;
  dim3 grid((channels + 31) / 32, numSegs);
  dim3 block(32, POOL_SLICES);
  sparseDotsPoolKernel<<<grid, block, 0, stream>>>(x, segStart, boardOffset, out, channels, valueHeadMode);
}

//------------------------------------------------------------------------------------------------------------------

// A block per row and 32 output channels, the sum over the inputs split over SMALL_MATMUL_SLICES rows of threads
constexpr int SMALL_MATMUL_SLICES = 8;

__global__ void sparseDotsSmallMatMulKernel(
  const float* in, const float* weight, const float* bias, float* out, int inC, int outC, int activation
) {
  __shared__ float partial[SMALL_MATMUL_SLICES][32];
  const int r = blockIdx.y;
  const int c = blockIdx.x * 32 + threadIdx.x;
  float acc = 0.0f;
  if(c < outC) {
    for(int k = threadIdx.y; k < inC; k += SMALL_MATMUL_SLICES)
      acc += in[(size_t)r * inC + k] * weight[(size_t)k * outC + c];
  }
  partial[threadIdx.y][threadIdx.x] = acc;
  __syncthreads();
  if(threadIdx.y == 0 && c < outC) {
    float sum = bias != nullptr ? bias[c] : 0.0f;
    for(int s = 0; s < SMALL_MATMUL_SLICES; s++)
      sum += partial[s][threadIdx.x];
    out[(size_t)r * outC + c] = activate(sum, activation);
  }
}

void sparseDotsSmallMatMul(
  const float* in, const float* weight, const float* bias, float* out, int rows, int inC, int outC, int activation,
  cudaStream_t stream
) {
  if(rows > 65535)
    throw std::runtime_error("sparseDotsSmallMatMul: too many rows");
  if(rows <= 0)
    return;
  dim3 grid((outC + 31) / 32, rows);
  dim3 block(32, SMALL_MATMUL_SLICES);
  sparseDotsSmallMatMulKernel<<<grid, block, 0, stream>>>(in, weight, bias, out, inC, outC, activation);
}

//------------------------------------------------------------------------------------------------------------------

#define SPARSE_DOTS_INSTANTIATE(T) \
  template void sparseDotsEmbedTokens<T>( \
    T*, const float*, const float*, const float*, int, const float*, const float*, const float*, int, const int*, \
    const int*, int, int, cudaStream_t); \
  template void sparseDotsEmbedQueries<T>( \
    T*, const float*, const float*, const float*, int, const float*, const float*, const float*, const float*, \
    const int*, const int*, int, int, cudaStream_t); \
  template void sparseDotsScaleBiasAct<T>(const T*, T*, const float*, const float*, long long, int, int, cudaStream_t); \
  template void sparseDotsBiasAct<T>(T*, const float*, long long, int, int, cudaStream_t); \
  template void sparseDotsRMSNorm<T>(const T*, T*, const float*, long long, int, float, cudaStream_t); \
  template void sparseDotsSwiGLU<T>(T*, long long, int, cudaStream_t); \
  template void sparseDotsSpatialRMSNorm<T>( \
    const T*, T*, float*, const float*, const float*, const int*, const int*, int, int, int, float, int, float*, \
    cudaStream_t); \
  template void sparseDotsApplyRope<T>(T*, const float*, const float*, int, int, int, int, int, cudaStream_t); \
  template void sparseDotsApplyRopeQK<T>(T*, T*, const float*, const float*, int, int, int, int, int, int, cudaStream_t); \
  template void sparseDotsAttention<T>( \
    const T*, const T*, const T*, T*, const int*, const int*, int, int, int, int, int, int, int, int, const int*, \
    const float*, cudaStream_t);

SPARSE_DOTS_INSTANTIATE(float)
SPARSE_DOTS_INSTANTIATE(half)
