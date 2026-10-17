// CUDA kernels of the sparse Dots models (model version 18), see cudasparsedots.h.

#include "../neuralnet/cudasparsedots.h"

#include <stdexcept>

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

//------------------------------------------------------------------------------------------------------------------

template <typename From, typename To>
__global__ void sparseDotsConvertKernel(const From* in, To* out, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
    out[i] = fromFloat<To>(toFloat(in[i]));
}

void sparseDotsCopyToHalf(const float* in, half* out, long long n, cudaStream_t stream) {
  if(n > 0)
    sparseDotsConvertKernel<float, half><<<numBlocksFor(n, 256), 256, 0, stream>>>(in, out, n);
}
void sparseDotsCopyFromHalf(const half* in, float* out, long long n, cudaStream_t stream) {
  if(n > 0)
    sparseDotsConvertKernel<half, float><<<numBlocksFor(n, 256), 256, 0, stream>>>(in, out, n);
}

//------------------------------------------------------------------------------------------------------------------

__global__ void sparseDotsScatterRowsKernel(const float* src, float* dst, const int* dstRows, int numRows, int channels) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)numRows * channels)
    return;
  int r = (int)(i / channels);
  int c = (int)(i % channels);
  dst[(long long)dstRows[r] * channels + c] = src[i];
}

void sparseDotsScatterRows(const float* src, float* dst, const int* dstRows, int numRows, int channels, cudaStream_t stream) {
  if(numRows <= 0)
    return;
  long long n = (long long)numRows * channels;
  sparseDotsScatterRowsKernel<<<numBlocksFor(n, 256), 256, 0, stream>>>(src, dst, dstRows, numRows, channels);
}

//------------------------------------------------------------------------------------------------------------------

__global__ void sparseDotsAddGlobalKernel(
  float* seq, const float* globalToken, const float* globalProj, const int* rowSeg, const int* segStart, int rows, int channels
) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)rows * channels)
    return;
  int c = (int)(i % channels);
  int r = (int)(i / channels);
  int n = rowSeg[r];
  float g = globalProj[(long long)n * channels + c];
  if(r == segStart[n])
    seq[i] = globalToken[c] + g;
  else
    seq[i] += g;
}

void sparseDotsAddGlobal(
  float* seq, const float* globalToken, const float* globalProj, const int* rowSeg, const int* segStart, int rows,
  int channels, cudaStream_t stream
) {
  long long n = (long long)rows * channels;
  if(n > 0)
    sparseDotsAddGlobalKernel<<<numBlocksFor(n, 256), 256, 0, stream>>>(seq, globalToken, globalProj, rowSeg, segStart, rows, channels);
}

__global__ void sparseDotsAddRowVectorKernel(float* x, const float* v, const int* rowSeg, int rows, int channels) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)rows * channels)
    return;
  int c = (int)(i % channels);
  int r = (int)(i / channels);
  x[i] += v[(long long)rowSeg[r] * channels + c];
}

void sparseDotsAddRowVector(float* x, const float* v, const int* rowSeg, int rows, int channels, cudaStream_t stream) {
  long long n = (long long)rows * channels;
  if(n > 0)
    sparseDotsAddRowVectorKernel<<<numBlocksFor(n, 256), 256, 0, stream>>>(x, v, rowSeg, rows, channels);
}

//------------------------------------------------------------------------------------------------------------------

template <typename T>
__global__ void sparseDotsScaleBiasActKernel(
  const T* in, T* out, const float* scale, const float* bias, long long rows, int channels, int activation
) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= rows * channels)
    return;
  int c = (int)(i % channels);
  out[i] = fromFloat<T>(activate(toFloat(in[i]) * scale[c] + bias[c], activation));
}

template <typename T>
void sparseDotsScaleBiasAct(
  const T* in, T* out, const float* scale, const float* bias, long long rows, int channels, int activation,
  cudaStream_t stream
) {
  long long n = rows * channels;
  if(n > 0)
    sparseDotsScaleBiasActKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(in, out, scale, bias, rows, channels, activation);
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
__global__ void sparseDotsRMSNormKernel(const T* in, T* out, const float* weight, long long rows, int channels, float epsilon) {
  long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  if(row >= rows)
    return;
  const T* x = in + row * channels;
  float sumSq = 0.0f;
  for(int c = lane; c < channels; c += 32) {
    float v = toFloat(x[c]);
    sumSq += v * v;
  }
  for(int offset = 16; offset > 0; offset /= 2)
    sumSq += __shfl_xor_sync(0xffffffff, sumSq, offset);
  float inv = rsqrtf(sumSq / (float)channels + epsilon);
  T* o = out + row * channels;
  for(int c = lane; c < channels; c += 32)
    o[c] = fromFloat<T>(toFloat(x[c]) * inv * weight[c]);
}

template <typename T>
void sparseDotsRMSNorm(const T* in, T* out, const float* weight, long long rows, int channels, float epsilon, cudaStream_t stream) {
  if(rows > 0)
    sparseDotsRMSNormKernel<T><<<numBlocksFor(rows * 32, 128), 128, 0, stream>>>(in, out, weight, rows, channels, epsilon);
}

template <typename T>
__global__ void sparseDotsResidualAddKernel(T* x, const T* delta, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
    x[i] = fromFloat<T>(toFloat(x[i]) + toFloat(delta[i]));
}

template <typename T>
void sparseDotsResidualAdd(T* x, const T* delta, long long rows, int channels, cudaStream_t stream) {
  long long n = rows * channels;
  if(n > 0)
    sparseDotsResidualAddKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(x, delta, n);
}

template <typename T>
__global__ void sparseDotsSwiGLUKernel(T* ab, long long rows, int channels) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= rows * channels)
    return;
  long long r = i / channels;
  int c = (int)(i % channels);
  T* a = ab + r * 2 * channels + c;
  float v = toFloat(*a);
  *a = fromFloat<T>(v / (1.0f + __expf(-v)) * toFloat(a[channels]));
}

template <typename T>
void sparseDotsSwiGLU(T* ab, long long rows, int channels, cudaStream_t stream) {
  long long n = rows * channels;
  if(n > 0)
    sparseDotsSwiGLUKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(ab, rows, channels);
}

//------------------------------------------------------------------------------------------------------------------

constexpr int SPATIAL_RMSNORM_BLOCKS_PER_SEG = 8;

template <typename T>
__global__ void sparseDotsSpatialSumSqKernel(const T* in, const int* segStart, int channels, float* sumSqBuf) {
  __shared__ float shared[256];
  int n = blockIdx.y;
  long long begin = (long long)segStart[n] * channels;
  long long end = (long long)segStart[n + 1] * channels;
  float acc = 0.0f;
  for(long long i = begin + (long long)blockIdx.x * blockDim.x + threadIdx.x; i < end; i += (long long)blockDim.x * gridDim.x) {
    float v = toFloat(in[i]);
    acc += v * v;
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
  const T* in, T* out, const float* gamma, const float* beta, const int* rowSeg, const int* segStart, int rows, int channels,
  float epsilon, int activation, const float* sumSqBuf
) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)rows * channels)
    return;
  int r = (int)(i / channels);
  int c = (int)(i % channels);
  int n = rowSeg[r];
  float numTokens = (float)(segStart[n + 1] - segStart[n]);
  float inv = rsqrtf(sumSqBuf[n] / (numTokens * (float)channels) + epsilon);
  out[i] = fromFloat<T>(activate(toFloat(in[i]) * inv * gamma[c] + beta[c], activation));
}

template <typename T>
void sparseDotsSpatialRMSNorm(
  const T* in, T* out, const float* gamma, const float* beta, const int* rowSeg, const int* segStart, int numSegs, int rows,
  int channels, float epsilon, int activation, float* sumSqBuf, cudaStream_t stream
) {
  if(numSegs > 65535)
    throw std::runtime_error("sparseDotsSpatialRMSNorm: too many positions");
  if(rows <= 0)
    return;
  cudaMemsetAsync(sumSqBuf, 0, sizeof(float) * numSegs, stream);
  dim3 grid(SPATIAL_RMSNORM_BLOCKS_PER_SEG, numSegs);
  sparseDotsSpatialSumSqKernel<T><<<grid, 256, 0, stream>>>(in, segStart, channels, sumSqBuf);
  long long n = (long long)rows * channels;
  sparseDotsSpatialApplyKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(
    in, out, gamma, beta, rowSeg, segStart, rows, channels, epsilon, activation, sumSqBuf);
}

//------------------------------------------------------------------------------------------------------------------

template <typename T>
__global__ void sparseDotsApplyRopeKernel(
  T* x, const float* freqs, const float* pos, int rows, int numHeads, int numKVHeads, int headDim, int rowStride
) {
  int numPairs = headDim / 2;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)rows * numHeads * numPairs)
    return;
  int p = (int)(i % numPairs);
  long long rh = i / numPairs;
  int h = (int)(rh % numHeads);
  long long r = rh / numHeads;
  int kvh = h * numKVHeads / numHeads;
  const float* rowPos = pos + r * 3;
  T* pair = x + r * rowStride + (long long)h * headDim + 2 * p;
  if(rowPos[2] == 0.0f) {
    pair[0] = fromFloat<T>(0.0f);
    pair[1] = fromFloat<T>(0.0f);
    return;
  }
  const float* f = freqs + 2 * ((long long)kvh * numPairs + p);
  float s, c;
  sincosf(rowPos[0] * f[0] + rowPos[1] * f[1], &s, &c);
  float x0 = toFloat(pair[0]);
  float x1 = toFloat(pair[1]);
  pair[0] = fromFloat<T>(x0 * c - x1 * s);
  pair[1] = fromFloat<T>(x0 * s + x1 * c);
}

template <typename T>
void sparseDotsApplyRope(
  T* x, const float* freqs, const float* pos, int rows, int numHeads, int numKVHeads, int headDim, int rowStride,
  cudaStream_t stream
) {
  if(rows <= 0)
    return;
  long long n = (long long)rows * numHeads * (headDim / 2);
  sparseDotsApplyRopeKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(x, freqs, pos, rows, numHeads, numKVHeads, headDim, rowStride);
}

template <typename T>
__device__ __forceinline__ void sparseDotsRotatePair(T* pair, float c, float s) {
  float x0 = toFloat(pair[0]);
  float x1 = toFloat(pair[1]);
  pair[0] = fromFloat<T>(x0 * c - x1 * s);
  pair[1] = fromFloat<T>(x0 * s + x1 * c);
}

template <typename T>
__global__ void sparseDotsApplyRopeQKKernel(
  T* q, T* k, const float* freqs, const float* pos, int rows, int numHeads, int numKVHeads, int headDim, int qStride,
  int kStride
) {
  int numPairs = headDim / 2;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)rows * numKVHeads * numPairs)
    return;
  int p = (int)(i % numPairs);
  long long rh = i / numPairs;
  int kvh = (int)(rh % numKVHeads);
  long long r = rh / numKVHeads;
  const float* rowPos = pos + r * 3;
  float s = 0.0f, c = 0.0f;
  if(rowPos[2] != 0.0f) {
    const float* f = freqs + 2 * ((long long)kvh * numPairs + p);
    sincosf(rowPos[0] * f[0] + rowPos[1] * f[1], &s, &c);
  }
  sparseDotsRotatePair(k + r * kStride + (long long)kvh * headDim + 2 * p, c, s);
  int rep = numHeads / numKVHeads;
  for(int j = 0; j < rep; j++)
    sparseDotsRotatePair(q + r * qStride + (long long)(kvh * rep + j) * headDim + 2 * p, c, s);
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
  long long n = (long long)rows * numKVHeads * (headDim / 2);
  sparseDotsApplyRopeQKKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(
    q, k, freqs, pos, rows, numHeads, numKVHeads, headDim, qStride, kStride);
}

//------------------------------------------------------------------------------------------------------------------
// Attention with online softmax, tiled like flashAttentionTiledImpl in cudaandrocmhelpers.inc but over the packed
// positions. One thread per query, BLOCK_KV keys in shared memory. Used when the tensor-core kernel isn't.

template <typename T, int qHeadDim, int vHeadDim, int BLOCK_Q, int BLOCK_KV>
__global__ void sparseDotsAttentionKernel(
  const T* Q, const T* K, const T* V, T* output, const int* qStart, const int* kvStart, int numHeads, int numKVHeads,
  int qStride, int kvStride, float scale
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

  for(int kvTileStart = 0; kvTileStart < kvLen; kvTileStart += BLOCK_KV) {
    for(int t = tid; t < BLOCK_KV * qHeadDim; t += BLOCK_Q) {
      int p = kvTileStart + t / qHeadDim;
      kTile[t] = p < kvLen ? toFloat(K[(long long)(kvBase + p) * kvStride + kvh * qHeadDim + t % qHeadDim]) : 0.0f;
    }
    for(int t = tid; t < BLOCK_KV * vHeadDim; t += BLOCK_Q) {
      int p = kvTileStart + t / vHeadDim;
      vTile[t] = p < kvLen ? toFloat(V[(long long)(kvBase + p) * kvStride + kvh * vHeadDim + t % vHeadDim]) : 0.0f;
    }
    __syncthreads();
    if(active) {
      int kvEnd = min(BLOCK_KV, kvLen - kvTileStart);
      for(int tk = 0; tk < kvEnd; tk++) {
        float dot = 0.0f;
        #pragma unroll
        for(int d = 0; d < qHeadDim; d++)
          dot += qReg[d] * kTile[tk * qHeadDim + d];
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
  int numHeads, int numKVHeads, int qHeadDim, int vHeadDim, int qStride, int kvStride, cudaStream_t stream
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
      Q, K, V, output, qStart, kvStart, numHeads, numKVHeads, qStride, kvStride, scale);
  else if(qHeadDim == 64 && vHeadDim == 64)
    sparseDotsAttentionKernel<T, 64, 64, BLOCK_Q, BLOCK_KV><<<grid, BLOCK_Q, 0, stream>>>(
      Q, K, V, output, qStart, kvStart, numHeads, numKVHeads, qStride, kvStride, scale);
  else if(qHeadDim == 16 && vHeadDim == 16)
    sparseDotsAttentionKernel<T, 16, 16, BLOCK_Q, BLOCK_KV><<<grid, BLOCK_Q, 0, stream>>>(
      Q, K, V, output, qStart, kvStart, numHeads, numKVHeads, qStride, kvStride, scale);
  else
    throw std::runtime_error("sparseDotsAttention: unsupported head dims");
}

//------------------------------------------------------------------------------------------------------------------

__global__ void sparseDotsPoolKernel(
  const float* x, const int* segStart, const float* boardOffset, float* out, int numSegs, int channels, bool valueHeadMode
) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)numSegs * channels)
    return;
  int n = (int)(i / channels);
  int c = (int)(i % channels);
  float sum = 0.0f;
  float maxV = -1e30f;
  const int begin = segStart[n];
  const int end = segStart[n + 1];
  for(int r = begin; r < end; r++) {
    float v = x[(long long)r * channels + c];
    sum += v;
    maxV = fmaxf(maxV, v);
  }
  float mean = sum / (float)(end - begin);
  float off = boardOffset[n];
  float* o = out + (long long)n * 3 * channels;
  o[c] = mean;
  o[channels + c] = mean * (off / 10.0f);
  o[2 * channels + c] = valueHeadMode ? mean * (off * off / 100.0f - 0.1f) : maxV;
}

void sparseDotsPool(
  const float* x, const int* segStart, const float* boardOffset, float* out, int numSegs, int channels, bool valueHeadMode,
  cudaStream_t stream
) {
  long long n = (long long)numSegs * channels;
  if(n > 0)
    sparseDotsPoolKernel<<<numBlocksFor(n, 128), 128, 0, stream>>>(x, segStart, boardOffset, out, numSegs, channels, valueHeadMode);
}

//------------------------------------------------------------------------------------------------------------------

#define SPARSE_DOTS_INSTANTIATE(T) \
  template void sparseDotsScaleBiasAct<T>(const T*, T*, const float*, const float*, long long, int, int, cudaStream_t); \
  template void sparseDotsBiasAct<T>(T*, const float*, long long, int, int, cudaStream_t); \
  template void sparseDotsRMSNorm<T>(const T*, T*, const float*, long long, int, float, cudaStream_t); \
  template void sparseDotsResidualAdd<T>(T*, const T*, long long, int, cudaStream_t); \
  template void sparseDotsSwiGLU<T>(T*, long long, int, cudaStream_t); \
  template void sparseDotsSpatialRMSNorm<T>( \
    const T*, T*, const float*, const float*, const int*, const int*, int, int, int, float, int, float*, cudaStream_t); \
  template void sparseDotsApplyRope<T>(T*, const float*, const float*, int, int, int, int, int, cudaStream_t); \
  template void sparseDotsApplyRopeQK<T>(T*, T*, const float*, const float*, int, int, int, int, int, int, cudaStream_t); \
  template void sparseDotsAttention<T>( \
    const T*, const T*, const T*, T*, const int*, const int*, int, int, int, int, int, int, int, int, cudaStream_t);

SPARSE_DOTS_INSTANTIATE(float)
SPARSE_DOTS_INSTANTIATE(half)
