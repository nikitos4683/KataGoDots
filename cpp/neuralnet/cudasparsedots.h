#ifndef NEURALNET_CUDASPARSEDOTS_H_
#define NEURALNET_CUDASPARSEDOTS_H_

// CUDA kernels of the sparse Dots models (model version 18), see docs/DotsSparseNN.md and cudasparsedotsmodel.h.
// The rows (tokens or candidate queries) of all the positions of a batch are packed without padding, as
// [rows, channels] row-major: position n owns the rows [segStart[n], segStart[n+1]), and rowSeg[r] is the position
// of row r. The trunk rows of a position are [GLOBAL][POINT x P], its query rows are [candidates][grounding].
// Activations T are float or half, parameters (scales, biases, norm weights) are always float, and all
// the arithmetic is in float. The elementwise kernels need the channels (and the row strides and head dims of RoPE)
// to be multiples of 8 and the rows 16-byte aligned.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// n elements, 16-byte aligned
void sparseDotsCopyFromHalf(const half* in, float* out, long long n, cudaStream_t stream);

// The trunk rows of the embedded tokens: for row r of position n = rowSeg[r], the first row of the position (GLOBAL)
// is globalToken + g[n], the other rows are the points (point i of the position at row segStart[n] + 1 + i, its
// features at pointFeat[r - n - 1]): pointFeat * weight + bias + g[n], where g[n] = globalFeat[n] * globalWeight.
// The weights are [features x channels] row-major.
template <typename T>
void sparseDotsEmbedTokens(
  T* out, const float* pointFeat, const float* weight, const float* bias, int numFeatures, const float* globalToken,
  const float* globalFeat, const float* globalWeight, int numGlobalFeatures, const int* rowSeg, const int* segStart,
  int rows, int channels, cudaStream_t stream);

// The decoder rows of the embedded queries: row r is candidate rowSrc[r] >= 0 (candFeat * candWeight + candBias) or
// grounding -rowSrc[r] - 1 (groundFeat * groundWeight + groundBias, one feature), plus poolProj[rowSeg[r]].
template <typename T>
void sparseDotsEmbedQueries(
  T* out, const float* candFeat, const float* candWeight, const float* candBias, int numCandFeatures,
  const float* groundFeat, const float* groundWeight, const float* groundBias, const float* poolProj, const int* rowSeg,
  const int* rowSrc, int rows, int channels, cudaStream_t stream);

// out = act(in * scale + bias)
template <typename T>
void sparseDotsScaleBiasAct(
  const T* in, T* out, const float* scale, const float* bias, long long rows, int channels, int activation,
  cudaStream_t stream);

// buf = act(buf + bias), bias may be null
template <typename T>
void sparseDotsBiasAct(T* buf, const float* bias, long long rows, int channels, int activation, cudaStream_t stream);

// Per-row RMSNorm: out = in / sqrt(mean(in^2) + epsilon) * weight
template <typename T>
void sparseDotsRMSNorm(const T* in, T* out, const float* weight, long long rows, int channels, float epsilon, cudaStream_t stream);

// ab [rows, 2 * channels] holds a then g in each row: a = silu(a) * g, in place (g is left as is)
template <typename T>
void sparseDotsSwiGLU(T* ab, long long rows, int channels, cudaStream_t stream);

// RMSNorm over all the rows and channels of each position, then out = act(x * inv * gamma + beta), also written
// to outFloat if not null (as out converted to float). sumSqBuf has numSegs floats. In place is fine.
template <typename T>
void sparseDotsSpatialRMSNorm(
  const T* in, T* out, float* outFloat, const float* gamma, const float* beta, const int* rowSeg, const int* segStart,
  int numSegs, int rows, int channels, float epsilon, int activation, float* sumSqBuf, cudaStream_t stream);

// Rotates the interleaved pairs (2p, 2p+1) of every head of x [rows, rowStride] (heads at the start of the row) by the
// angles x_r * fx + y_r * fy, where pos [rows, 3] holds (x, y, hasPosition) and freqs [numKVHeads * numPairs, 2]
// (fx, fy); head h uses kv head h * numKVHeads / numHeads. Rows without a position become zero (NoPE).
template <typename T>
void sparseDotsApplyRope(
  T* x, const float* freqs, const float* pos, int rows, int numHeads, int numKVHeads, int headDim, int rowStride,
  cudaStream_t stream);

// The same for the queries and the keys of the same rows (self-attention), computing each angle once:
// q [rows, qStride] has numHeads heads, k [rows, kStride] has numKVHeads heads. numHeads must divide by numKVHeads.
template <typename T>
void sparseDotsApplyRopeQK(
  T* q, T* k, const float* freqs, const float* pos, int rows, int numHeads, int numKVHeads, int headDim, int qStride,
  int kStride, cudaStream_t stream);

// Attention of the queries of every position to the keys of the same position: queries Q [qRows, qStride] (heads
// at the start of the row), keys K and values V [kvRows, kvStride], output [qRows, numHeads * vHeadDim].
// Position n has the queries [qStart[n], qStart[n+1]) and the keys [kvStart[n], kvStart[n+1]); maxQLen is the
// max number of queries of a position. Head dims (16, 16), (32, 32) and (64, 64) are supported.
// chainCodes/chainBias (self-attention of the trunk, both null if none): a code per row and the bias [numHeads][3] of
// the pairs of rows by their codes, see SparseDotsNN::chainCategory.
template <typename T>
void sparseDotsAttention(
  const T* Q, const T* K, const T* V, T* output, const int* qStart, const int* kvStart, int numSegs, int maxQLen,
  int numHeads, int numKVHeads, int qHeadDim, int vHeadDim, int qStride, int kvStride, const int* chainCodes,
  const float* chainBias, cudaStream_t stream);

// Pooling over the rows of each position of x [rows, channels], out [numSegs, 3 * channels]:
// gpool mode:  [mean, mean * off / 10, max]
// value mode:  [mean, mean * off / 10, mean * (off * off / 100 - 0.1)]
// where off = boardOffset[n] = sqrt(xSize * ySize) - 14.
void sparseDotsPool(
  const float* x, const int* segStart, const float* boardOffset, float* out, int numSegs, int channels, bool valueHeadMode,
  cudaStream_t stream);

// out [rows, outC] = act(in [rows, inC] * weight [inC x outC] + bias), FP32, bias may be null. For the few rows of
// the positions (one per position), where cuBLAS has too little parallelism.
void sparseDotsSmallMatMul(
  const float* in, const float* weight, const float* bias, float* out, int rows, int inC, int outC, int activation,
  cudaStream_t stream);

#endif  // NEURALNET_CUDASPARSEDOTS_H_
