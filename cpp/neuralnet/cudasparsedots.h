#ifndef NEURALNET_CUDASPARSEDOTS_H_
#define NEURALNET_CUDASPARSEDOTS_H_

// CUDA kernels of the sparse Dots models (model version 18), see docs/DotsSparseNN.md and cudasparsedotsmodel.h.
// Token sequences are laid out as [batch, seqLen, channels] (row-major), the trunk tokens of a row being
// [GLOBAL][POINT x P][AREA x A] followed by padding. Positions of tokens for RoPE are weighted sets given by
// segments: token t owns the entries [segStart[t], segStart[t+1]) of posXYW (x, y, weight).
// Activations T are float or half, parameters (scales, biases, norm weights, masks) are always float, and all
// the arithmetic is in float. A mask (may be null where noted) has one value per row (token).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void sparseDotsCopyToHalf(const float* in, half* out, long long n, cudaStream_t stream);
void sparseDotsCopyFromHalf(const half* in, float* out, long long n, cudaStream_t stream);

// dst[dstRows[r], :] = src[r, :] for r < numRows
void sparseDotsScatterRows(const float* src, float* dst, const int* dstRows, int numRows, int channels, cudaStream_t stream);

// For every row n and token s: the GLOBAL token (s == 0) becomes globalToken + globalProj[n], the other valid
// tokens get globalProj[n] added, padding becomes 0.
void sparseDotsAddGlobal(
  float* seq, const float* globalToken, const float* globalProj, const float* mask, int batchSize, int seqLen, int channels,
  cudaStream_t stream);

// x[n, l, :] += v[n, :]
void sparseDotsAddRowVector(float* x, const float* v, int batchSize, int len, int channels, cudaStream_t stream);

// For every segment a with entries [segStart[a], segStart[a+1]) of h [entries, channels]:
// out[a, 0:channels] = mean, out[a, channels:2*channels] = max (zeros for an empty segment).
void sparseDotsSegmentMeanMax(
  const float* h, const int* segStart, int numSegments, int channels, float* out, int outStride, cudaStream_t stream);

// out = act(in * scale + bias) * mask
template <typename T>
void sparseDotsScaleBiasAct(
  const T* in, T* out, const float* scale, const float* bias, const float* mask, long long rows, int channels, int activation,
  cudaStream_t stream);

// buf = act(buf + bias), bias may be null
template <typename T>
void sparseDotsBiasAct(T* buf, const float* bias, long long rows, int channels, int activation, cudaStream_t stream);

// Per-row RMSNorm: out = in / sqrt(mean(in^2) + epsilon) * weight (* mask if not null)
template <typename T>
void sparseDotsRMSNorm(
  const T* in, T* out, const float* weight, const float* mask, long long rows, int channels, float epsilon,
  cudaStream_t stream);

// dst[dstRows[r], :] += src[r, :] for r < numRows
template <typename T>
void sparseDotsScatterAddRows(const T* src, T* dst, const int* dstRows, int numRows, int channels, cudaStream_t stream);

// x += delta * mask (mask may be null)
template <typename T>
void sparseDotsResidualAdd(T* x, const T* delta, const float* mask, long long rows, int channels, cudaStream_t stream);

// a = silu(a) * g
template <typename T>
void sparseDotsSwiGLU(T* a, const T* g, long long n, cudaStream_t stream);

// RMSNorm over all the valid tokens and channels of each row n of x [batch, seqLen, channels], then
// out = act(x * inv * gamma + beta) * mask. sumSqBuf has batchSize floats. In place is fine.
template <typename T>
void sparseDotsSpatialRMSNorm(
  const T* in, T* out, const float* gamma, const float* beta, const float* mask, const float* numTokens,
  int batchSize, int seqLen, int channels, float epsilon, int activation, float* sumSqBuf, cudaStream_t stream);

// cos/sin tables [numTokens, numKVHeads * numPairs] for the frequencies freqs [numKVHeads * numPairs, 2]:
// cos[t, i] = sum over the entries e of t of w_e * cos(x_e * fx_i + y_e * fy_i), the same for sin.
void sparseDotsRopeTables(
  const float* freqs, const int* segStart, const float* posXYW, int numTokens, int numKVHeadPairs, float* cosOut, float* sinOut,
  cudaStream_t stream);

// Rotates the interleaved pairs (2p, 2p+1) of every head of x [numTokens, rowStride] (heads at the start of the row)
// with the tables [numTokens, numKVHeads * numPairs]; head h uses kv head h * numKVHeads / numHeads.
template <typename T>
void sparseDotsApplyRope(
  T* x, const float* cosTable, const float* sinTable, int numTokens, int numHeads, int numKVHeads, int headDim, int rowStride,
  cudaStream_t stream);

// Attention of the queries Q [batch, qLen, numHeads, qHeadDim] to the keys K [batch, kvLen, numKVHeads, qHeadDim]
// and values V [batch, kvLen, numKVHeads, vHeadDim] where only the keys with kvMask [batch, kvLen] != 0 count.
// Output [batch, qLen, numHeads, vHeadDim]. All rows are dense (row stride = heads * dim).
// Head dims (16, 16), (32, 32) and (64, 64) are supported.
template <typename T>
void sparseDotsAttention(
  const T* Q, const T* K, const T* V, const float* kvMask, T* output,
  int batchSize, int qLen, int kvLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim, cudaStream_t stream);

// Attention of each area to the points of its boundary only. Q [tokens, numHeads, qHeadDim],
// K [tokens, numKVHeads, qHeadDim], V [tokens, numKVHeads, vHeadDim] (dense rows). Area a is the token areaRows[a],
// its boundary is the tokens vertexRows[e] for e in [segStart[a], segStart[a+1]).
// Output [numAreas, numHeads, vHeadDim], zeros for an area without vertices.
// Head dims (16, 16), (32, 32) and (64, 64) are supported.
template <typename T>
void sparseDotsBoundaryAttention(
  const T* Q, const T* K, const T* V, const int* areaRows, const int* segStart, const int* vertexRows, T* output,
  int numAreas, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim, cudaStream_t stream);

// Pooling over the valid tokens of x [batch, seqLen, channels], out [batch, 3 * channels]:
// gpool mode:  [mean, mean * off / 10, max]
// value mode:  [mean, mean * off / 10, mean * (off * off / 100 - 0.1)]
// where mean = sum / numTokens[n] and off = boardOffset[n] = sqrt(xSize * ySize) - 14.
void sparseDotsPool(
  const float* x, const float* mask, const float* numTokens, const float* boardOffset, float* out,
  int batchSize, int seqLen, int channels, bool valueHeadMode, cudaStream_t stream);

#endif  // NEURALNET_CUDASPARSEDOTS_H_
