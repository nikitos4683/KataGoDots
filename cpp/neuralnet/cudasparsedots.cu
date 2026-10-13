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
  float* seq, const float* globalToken, const float* globalProj, const float* mask, int batchSize, int seqLen, int channels
) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)batchSize * seqLen * channels)
    return;
  int c = (int)(i % channels);
  long long ns = i / channels;
  int s = (int)(ns % seqLen);
  int n = (int)(ns / seqLen);
  float g = globalProj[(long long)n * channels + c];
  if(s == 0)
    seq[i] = globalToken[c] + g;
  else if(mask[ns] != 0.0f)
    seq[i] += g;
  else
    seq[i] = 0.0f;
}

void sparseDotsAddGlobal(
  float* seq, const float* globalToken, const float* globalProj, const float* mask, int batchSize, int seqLen, int channels,
  cudaStream_t stream
) {
  long long n = (long long)batchSize * seqLen * channels;
  sparseDotsAddGlobalKernel<<<numBlocksFor(n, 256), 256, 0, stream>>>(seq, globalToken, globalProj, mask, batchSize, seqLen, channels);
}

__global__ void sparseDotsAddRowVectorKernel(float* x, const float* v, int batchSize, int len, int channels) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)batchSize * len * channels)
    return;
  int c = (int)(i % channels);
  int n = (int)(i / ((long long)len * channels));
  x[i] += v[(long long)n * channels + c];
}

void sparseDotsAddRowVector(float* x, const float* v, int batchSize, int len, int channels, cudaStream_t stream) {
  long long n = (long long)batchSize * len * channels;
  if(n > 0)
    sparseDotsAddRowVectorKernel<<<numBlocksFor(n, 256), 256, 0, stream>>>(x, v, batchSize, len, channels);
}

//------------------------------------------------------------------------------------------------------------------

template <typename T>
__global__ void sparseDotsScaleBiasActKernel(
  const T* in, T* out, const float* scale, const float* bias, const float* mask, long long rows, int channels, int activation
) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= rows * channels)
    return;
  int c = (int)(i % channels);
  float m = mask != nullptr ? mask[i / channels] : 1.0f;
  out[i] = fromFloat<T>(activate(toFloat(in[i]) * scale[c] + bias[c], activation) * m);
}

template <typename T>
void sparseDotsScaleBiasAct(
  const T* in, T* out, const float* scale, const float* bias, const float* mask, long long rows, int channels, int activation,
  cudaStream_t stream
) {
  long long n = rows * channels;
  if(n > 0)
    sparseDotsScaleBiasActKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(in, out, scale, bias, mask, rows, channels, activation);
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
  const T* in, T* out, const float* weight, const float* mask, long long rows, int channels, float epsilon
) {
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
  if(mask != nullptr)
    inv *= mask[row];
  T* o = out + row * channels;
  for(int c = lane; c < channels; c += 32)
    o[c] = fromFloat<T>(toFloat(x[c]) * inv * weight[c]);
}

template <typename T>
void sparseDotsRMSNorm(
  const T* in, T* out, const float* weight, const float* mask, long long rows, int channels, float epsilon,
  cudaStream_t stream
) {
  if(rows > 0)
    sparseDotsRMSNormKernel<T><<<numBlocksFor(rows * 32, 128), 128, 0, stream>>>(in, out, weight, mask, rows, channels, epsilon);
}

template <typename T>
__global__ void sparseDotsResidualAddKernel(T* x, const T* delta, const float* mask, long long rows, int channels) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= rows * channels)
    return;
  float m = mask != nullptr ? mask[i / channels] : 1.0f;
  x[i] = fromFloat<T>(toFloat(x[i]) + toFloat(delta[i]) * m);
}

template <typename T>
void sparseDotsResidualAdd(T* x, const T* delta, const float* mask, long long rows, int channels, cudaStream_t stream) {
  long long n = rows * channels;
  if(n > 0)
    sparseDotsResidualAddKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(x, delta, mask, rows, channels);
}

template <typename T>
__global__ void sparseDotsSwiGLUKernel(T* a, const T* g, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= n)
    return;
  float v = toFloat(a[i]);
  a[i] = fromFloat<T>(v / (1.0f + __expf(-v)) * toFloat(g[i]));
}

template <typename T>
void sparseDotsSwiGLU(T* a, const T* g, long long n, cudaStream_t stream) {
  if(n > 0)
    sparseDotsSwiGLUKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(a, g, n);
}

//------------------------------------------------------------------------------------------------------------------

constexpr int SPATIAL_RMSNORM_BLOCKS_PER_ROW = 16;

template <typename T>
__global__ void sparseDotsSpatialSumSqKernel(const T* in, const float* mask, int seqLen, int channels, float* sumSqBuf) {
  __shared__ float shared[256];
  int n = blockIdx.y;
  long long total = (long long)seqLen * channels;
  const T* x = in + (long long)n * total;
  float acc = 0.0f;
  for(long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)blockDim.x * gridDim.x) {
    float v = toFloat(x[i]) * mask[(long long)n * seqLen + i / channels];
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
  const T* in, T* out, const float* gamma, const float* beta, const float* mask, const float* numTokens,
  int batchSize, int seqLen, int channels, float epsilon, int activation, const float* sumSqBuf
) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)batchSize * seqLen * channels)
    return;
  long long token = i / channels;
  int c = (int)(i % channels);
  int n = (int)(token / seqLen);
  float inv = rsqrtf(sumSqBuf[n] / (numTokens[n] * (float)channels) + epsilon);
  float m = mask[token];
  out[i] = fromFloat<T>(activate(toFloat(in[i]) * m * inv * gamma[c] + beta[c], activation) * m);
}

template <typename T>
void sparseDotsSpatialRMSNorm(
  const T* in, T* out, const float* gamma, const float* beta, const float* mask, const float* numTokens,
  int batchSize, int seqLen, int channels, float epsilon, int activation, float* sumSqBuf, cudaStream_t stream
) {
  if(batchSize > 65535)
    throw std::runtime_error("sparseDotsSpatialRMSNorm: batchSize too large");
  cudaMemsetAsync(sumSqBuf, 0, sizeof(float) * batchSize, stream);
  dim3 grid(SPATIAL_RMSNORM_BLOCKS_PER_ROW, batchSize);
  sparseDotsSpatialSumSqKernel<T><<<grid, 256, 0, stream>>>(in, mask, seqLen, channels, sumSqBuf);
  long long n = (long long)batchSize * seqLen * channels;
  sparseDotsSpatialApplyKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(
    in, out, gamma, beta, mask, numTokens, batchSize, seqLen, channels, epsilon, activation, sumSqBuf);
}

//------------------------------------------------------------------------------------------------------------------

__global__ void sparseDotsRopeTablesKernel(
  const float* freqs, const int* segStart, const float* posXYW, int numTokens, int numKVHeadPairs, float* cosOut, float* sinOut
) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)numTokens * numKVHeadPairs)
    return;
  int t = (int)(i / numKVHeadPairs);
  int j = (int)(i % numKVHeadPairs);
  float fx = freqs[2 * j];
  float fy = freqs[2 * j + 1];
  float c = 0.0f;
  float s = 0.0f;
  int end = segStart[t + 1];
  for(int e = segStart[t]; e < end; e++) {
    float x = posXYW[3 * e];
    float y = posXYW[3 * e + 1];
    float w = posXYW[3 * e + 2];
    float sn, cs;
    sincosf(x * fx + y * fy, &sn, &cs);
    c += w * cs;
    s += w * sn;
  }
  cosOut[i] = c;
  sinOut[i] = s;
}

void sparseDotsRopeTables(
  const float* freqs, const int* segStart, const float* posXYW, int numTokens, int numKVHeadPairs, float* cosOut, float* sinOut,
  cudaStream_t stream
) {
  if(numTokens <= 0)
    return;
  long long n = (long long)numTokens * numKVHeadPairs;
  sparseDotsRopeTablesKernel<<<numBlocksFor(n, 256), 256, 0, stream>>>(freqs, segStart, posXYW, numTokens, numKVHeadPairs, cosOut, sinOut);
}

template <typename T>
__global__ void sparseDotsApplyRopeKernel(
  T* x, const float* cosTable, const float* sinTable, int numTokens, int numHeads, int numKVHeads, int headDim, int rowStride
) {
  int numPairs = headDim / 2;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)numTokens * numHeads * numPairs)
    return;
  int p = (int)(i % numPairs);
  long long th = i / numPairs;
  int h = (int)(th % numHeads);
  long long t = th / numHeads;
  int kvh = h * numKVHeads / numHeads;
  long long tableIdx = t * numKVHeads * numPairs + (long long)kvh * numPairs + p;
  float c = cosTable[tableIdx];
  float s = sinTable[tableIdx];
  T* pair = x + t * rowStride + (long long)h * headDim + 2 * p;
  float x0 = toFloat(pair[0]);
  float x1 = toFloat(pair[1]);
  pair[0] = fromFloat<T>(x0 * c - x1 * s);
  pair[1] = fromFloat<T>(x0 * s + x1 * c);
}

template <typename T>
void sparseDotsApplyRope(
  T* x, const float* cosTable, const float* sinTable, int numTokens, int numHeads, int numKVHeads, int headDim, int rowStride,
  cudaStream_t stream
) {
  if(numTokens <= 0)
    return;
  long long n = (long long)numTokens * numHeads * (headDim / 2);
  sparseDotsApplyRopeKernel<T><<<numBlocksFor(n, 256), 256, 0, stream>>>(x, cosTable, sinTable, numTokens, numHeads, numKVHeads, headDim, rowStride);
}

//------------------------------------------------------------------------------------------------------------------
// Attention with online softmax, tiled like flashAttentionTiledImpl in cudaandrocmhelpers.inc but with different
// query and key lengths and no query mask. One thread per query, BLOCK_KV keys in shared memory.

template <typename T, int qHeadDim, int vHeadDim, int BLOCK_Q, int BLOCK_KV>
__global__ void sparseDotsAttentionKernel(
  const T* Q, const T* K, const T* V, const float* kvMask, T* output,
  int qLen, int kvLen, int numHeads, int numKVHeads, float scale
) {
  const int tid = threadIdx.x;
  const int qPos = blockIdx.x * BLOCK_Q + tid;
  const int bh = blockIdx.y;
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h * numKVHeads / numHeads;
  const int qTotalDim = numHeads * qHeadDim;
  const int kTotalDim = numKVHeads * qHeadDim;
  const int vTotalDim = numKVHeads * vHeadDim;
  const int oTotalDim = numHeads * vHeadDim;

  __shared__ float kTile[BLOCK_KV * qHeadDim];
  __shared__ float vTile[BLOCK_KV * vHeadDim];
  __shared__ float maskTile[BLOCK_KV];

  float qReg[qHeadDim];
  float acc[vHeadDim];
  float runningMax = -1e30f;
  float runningSum = 0.0f;
  const bool active = qPos < qLen;
  if(active) {
    const T* qPtr = Q + ((long long)n * qLen + qPos) * qTotalDim + h * qHeadDim;
    #pragma unroll
    for(int d = 0; d < qHeadDim; d++)
      qReg[d] = toFloat(qPtr[d]) * scale;
  }
  #pragma unroll
  for(int d = 0; d < vHeadDim; d++)
    acc[d] = 0.0f;

  for(int kvStart = 0; kvStart < kvLen; kvStart += BLOCK_KV) {
    for(int t = tid; t < BLOCK_KV * qHeadDim; t += BLOCK_Q) {
      int pos = kvStart + t / qHeadDim;
      kTile[t] = pos < kvLen ? toFloat(K[((long long)n * kvLen + pos) * kTotalDim + kvh * qHeadDim + t % qHeadDim]) : 0.0f;
    }
    for(int t = tid; t < BLOCK_KV * vHeadDim; t += BLOCK_Q) {
      int pos = kvStart + t / vHeadDim;
      vTile[t] = pos < kvLen ? toFloat(V[((long long)n * kvLen + pos) * vTotalDim + kvh * vHeadDim + t % vHeadDim]) : 0.0f;
    }
    for(int t = tid; t < BLOCK_KV; t += BLOCK_Q) {
      int pos = kvStart + t;
      maskTile[t] = pos < kvLen ? kvMask[(long long)n * kvLen + pos] : 0.0f;
    }
    __syncthreads();
    if(active) {
      int kvEnd = min(BLOCK_KV, kvLen - kvStart);
      for(int tk = 0; tk < kvEnd; tk++) {
        if(maskTile[tk] == 0.0f)
          continue;
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
    T* outRow = output + ((long long)n * qLen + qPos) * oTotalDim + h * vHeadDim;
    float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
    #pragma unroll
    for(int d = 0; d < vHeadDim; d++)
      outRow[d] = fromFloat<T>(acc[d] * invSum);
  }
}

template <typename T>
void sparseDotsAttention(
  const T* Q, const T* K, const T* V, const float* kvMask, T* output,
  int batchSize, int qLen, int kvLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim, cudaStream_t stream
) {
  if(batchSize * numHeads > 65535)
    throw std::runtime_error("sparseDotsAttention: batchSize * numHeads too large");
  if(qLen <= 0)
    return;
  float scale = 1.0f / sqrtf((float)qHeadDim);
  constexpr int BLOCK_Q = 64;
  constexpr int BLOCK_KV = 32;
  dim3 grid((qLen + BLOCK_Q - 1) / BLOCK_Q, batchSize * numHeads);
  if(qHeadDim == 32 && vHeadDim == 32)
    sparseDotsAttentionKernel<T, 32, 32, BLOCK_Q, BLOCK_KV><<<grid, BLOCK_Q, 0, stream>>>(Q, K, V, kvMask, output, qLen, kvLen, numHeads, numKVHeads, scale);
  else if(qHeadDim == 64 && vHeadDim == 64)
    sparseDotsAttentionKernel<T, 64, 64, BLOCK_Q, BLOCK_KV><<<grid, BLOCK_Q, 0, stream>>>(Q, K, V, kvMask, output, qLen, kvLen, numHeads, numKVHeads, scale);
  else if(qHeadDim == 16 && vHeadDim == 16)
    sparseDotsAttentionKernel<T, 16, 16, BLOCK_Q, BLOCK_KV><<<grid, BLOCK_Q, 0, stream>>>(Q, K, V, kvMask, output, qLen, kvLen, numHeads, numKVHeads, scale);
  else
    throw std::runtime_error("sparseDotsAttention: unsupported head dims");
}

//------------------------------------------------------------------------------------------------------------------

__global__ void sparseDotsPoolKernel(
  const float* x, const float* mask, const float* numTokens, const float* boardOffset, float* out,
  int batchSize, int seqLen, int channels, bool valueHeadMode
) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)batchSize * channels)
    return;
  int n = (int)(i / channels);
  int c = (int)(i % channels);
  float sum = 0.0f;
  float maxV = -1e30f;
  for(int s = 0; s < seqLen; s++) {
    if(mask[(long long)n * seqLen + s] == 0.0f)
      continue;
    float v = x[((long long)n * seqLen + s) * channels + c];
    sum += v;
    maxV = fmaxf(maxV, v);
  }
  float mean = sum / numTokens[n];
  float off = boardOffset[n];
  float* o = out + (long long)n * 3 * channels;
  o[c] = mean;
  o[channels + c] = mean * (off / 10.0f);
  o[2 * channels + c] = valueHeadMode ? mean * (off * off / 100.0f - 0.1f) : maxV;
}

void sparseDotsPool(
  const float* x, const float* mask, const float* numTokens, const float* boardOffset, float* out,
  int batchSize, int seqLen, int channels, bool valueHeadMode, cudaStream_t stream
) {
  long long n = (long long)batchSize * channels;
  sparseDotsPoolKernel<<<numBlocksFor(n, 128), 128, 0, stream>>>(x, mask, numTokens, boardOffset, out, batchSize, seqLen, channels, valueHeadMode);
}

//------------------------------------------------------------------------------------------------------------------

#define SPARSE_DOTS_INSTANTIATE(T) \
  template void sparseDotsScaleBiasAct<T>(const T*, T*, const float*, const float*, const float*, long long, int, int, cudaStream_t); \
  template void sparseDotsBiasAct<T>(T*, const float*, long long, int, int, cudaStream_t); \
  template void sparseDotsRMSNorm<T>(const T*, T*, const float*, const float*, long long, int, float, cudaStream_t); \
  template void sparseDotsResidualAdd<T>(T*, const T*, const float*, long long, int, cudaStream_t); \
  template void sparseDotsSwiGLU<T>(T*, const T*, long long, cudaStream_t); \
  template void sparseDotsSpatialRMSNorm<T>( \
    const T*, T*, const float*, const float*, const float*, const float*, int, int, int, float, int, float*, cudaStream_t); \
  template void sparseDotsApplyRope<T>(T*, const float*, const float*, int, int, int, int, int, cudaStream_t); \
  template void sparseDotsAttention<T>( \
    const T*, const T*, const T*, const float*, T*, int, int, int, int, int, int, int, cudaStream_t);

SPARSE_DOTS_INSTANTIATE(float)
SPARSE_DOTS_INSTANTIATE(half)
