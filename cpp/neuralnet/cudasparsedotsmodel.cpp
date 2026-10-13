#include "../neuralnet/cudasparsedotsmodel.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <type_traits>

#include "../core/logger.h"
#include "../core/test.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/cudasparsedots.h"
#include "../neuralnet/cudautils.h"

using namespace std;

namespace {

constexpr int SEQ_LEN_ALIGN = 8;

// A device buffer that only grows
struct DeviceBuf {
  void* ptr = nullptr;
  size_t bytes = 0;

  DeviceBuf() {}
  ~DeviceBuf() {
    if(ptr != nullptr)
      (void)cudaFree(ptr);
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;

  // Grows the buffer to at least the given size, dropping the content
  void ensure(size_t need) {
    if(need <= bytes)
      return;
    const size_t newBytes = std::max(need, bytes + bytes / 2);
    if(ptr != nullptr)
      (void)cudaFree(ptr);
    ptr = nullptr;
    bytes = 0;
    CUDA_ERR("SparseDotsCudaModel", cudaMalloc(&ptr, newBytes));
    bytes = newBytes;
  }
  void upload(const string& name, const vector<float>& values, bool useFP16) {
    testAssert(ptr == nullptr);
    if(values.empty())
      return;
    CudaUtils::mallocAndCopyToDevice(name, values, ptr, useFP16);
    bytes = values.size() * (useFP16 ? sizeof(half) : sizeof(float));
  }
  float* f() const { return (float*)ptr; }
  int* i() const { return (int*)ptr; }
};

// Weights [in x out] row-major, an optional FP32 bias
struct Linear {
  string name;
  int inC = 0;
  int outC = 0;
  bool useFP16 = false;
  DeviceBuf w;
  DeviceBuf b;

  void init(const MatMulLayerDesc& desc, const MatBiasLayerDesc* bias, bool fp16) {
    name = desc.name;
    inC = desc.inChannels;
    outC = desc.outChannels;
    useFP16 = fp16;
    testAssert(desc.weights.size() == (size_t)inC * outC);
    w.upload(name, desc.weights, useFP16);
    if(bias != nullptr) {
      testAssert(bias->numChannels == outC);
      b.upload(bias->name, bias->weights, false);
    }
  }
  void initConv1x1(const ConvLayerDesc& desc, bool fp16) {
    testAssert(desc.convXSize == 1 && desc.convYSize == 1);
    name = desc.name;
    inC = desc.inChannels;
    outC = desc.outChannels;
    useFP16 = fp16;
    // ConvLayerDesc stores [oc][ic]
    vector<float> transposed((size_t)inC * outC);
    for(int ic = 0; ic < inC; ic++)
      for(int oc = 0; oc < outC; oc++)
        transposed[(size_t)ic * outC + oc] = desc.weights[(size_t)oc * inC + ic];
    w.upload(name, transposed, useFP16);
  }
};

struct Attention {
  int numHeads = 0;
  int numKVHeads = 0;
  int qHeadDim = 0;
  int vHeadDim = 0;
  float epsilon = 0.0f;
  DeviceBuf normWeight;
  Linear q, k, v, out;
  DeviceBuf ropeFreqs;

  void init(const TransformerAttentionDesc& desc, bool fp16) {
    testAssert(desc.useRope && desc.learnableRope);
    numHeads = desc.numHeads;
    numKVHeads = desc.numKVHeads;
    qHeadDim = desc.qHeadDim;
    vHeadDim = desc.vHeadDim;
    epsilon = desc.preLN.epsilon;
    testAssert(qHeadDim % 2 == 0);
    testAssert(desc.ropeFreqs.size() == (size_t)numKVHeads * (qHeadDim / 2) * 2);
    normWeight.upload(desc.preLN.name, desc.preLN.weight, false);
    q.init(desc.qProj, nullptr, fp16);
    k.init(desc.kProj, nullptr, fp16);
    v.init(desc.vProj, nullptr, fp16);
    out.init(desc.outProj, nullptr, fp16);
    ropeFreqs.upload(desc.name + "/ropeFreqs", desc.ropeFreqs, false);
  }
  int numKVHeadPairs() const { return numKVHeads * (qHeadDim / 2); }
};

struct FFN {
  float epsilon = 0.0f;
  DeviceBuf normWeight;
  Linear linear1, gate, linear2;

  void init(const TransformerFFNDesc& desc, bool fp16) {
    testAssert(desc.useSwiGLU);
    epsilon = desc.preLN.epsilon;
    normWeight.upload(desc.preLN.name, desc.preLN.weight, false);
    linear1.init(desc.linear1, nullptr, fp16);
    gate.init(desc.linearGate, nullptr, fp16);
    linear2.init(desc.linear2, nullptr, fp16);
  }
};

struct TrunkBlock {
  DeviceBuf preScale, preBias;
  int preActivation = ACTIVATION_IDENTITY;
  Linear preConv;
  // Alternating attention and FFN sub-blocks
  vector<unique_ptr<Attention>> attentions;
  vector<unique_ptr<FFN>> ffns;
  vector<bool> isAttention;
  // Output projections of the area boundary attention, one per attention sub-block (empty if disabled)
  vector<unique_ptr<Linear>> boundaryOut;
  DeviceBuf postScale, postBias;
  int postActivation = ACTIVATION_IDENTITY;
  Linear postConv;
};

// Host staging of all the inputs of a batch, uploaded with a single copy
struct HostPack {
  vector<char> data;
  size_t size = 0;
  size_t reserve(size_t bytes) {
    const size_t offset = (size + 255) & ~(size_t)255;
    size = offset + bytes;
    return offset;
  }
};

}  // namespace

struct SparseDotsCudaModel::Impl {
  const bool useFP16;
  const bool useMmaAttention;
  cudaStream_t stream;
  cublasHandle_t cublas;
  Logger* logger;
  bool loggedMma = false;

  // Stage timing for development, enabled by the environment variable KATAGO_SPARSE_DOTS_PROFILE
  const bool profiling = std::getenv("KATAGO_SPARSE_DOTS_PROFILE") != nullptr;
  vector<cudaEvent_t> profileEvents;
  vector<string> profileNames;
  map<string, double> profileMs;
  int profileCalls = 0;
  int64_t profileRows = 0;

  void mark(const char* name) {
    if(!profiling)
      return;
    cudaEvent_t event;
    CUDA_ERR("SparseDotsCudaModel", cudaEventCreate(&event));
    CUDA_ERR("SparseDotsCudaModel", cudaEventRecord(event, stream));
    profileEvents.push_back(event);
    profileNames.push_back(name);
  }
  void finishProfile(double hostMs, int rows) {
    if(!profiling)
      return;
    for(size_t i = 1; i < profileEvents.size(); i++) {
      float ms = 0.0f;
      CUDA_ERR("SparseDotsCudaModel", cudaEventElapsedTime(&ms, profileEvents[i - 1], profileEvents[i]));
      profileMs[profileNames[i]] += ms;
    }
    for(cudaEvent_t event : profileEvents)
      (void)cudaEventDestroy(event);
    profileEvents.clear();
    profileNames.clear();
    profileMs["host pack"] += hostMs;
    profileCalls++;
    profileRows += rows;
    if(profileCalls % 50 == 0) {
      cout << "Sparse Dots CUDA profile, " << profileCalls << " calls, " << profileRows << " rows, ms per call:";
      for(const auto& entry : profileMs)
        cout << " " << entry.first << " " << entry.second / profileCalls;
      cout << endl;
    }
  }

  bool areaRopeMean;
  int trunkChannels;
  int midChannels;
  int vertexChannels;
  int decoderChannels;

  Linear point;
  Linear vertex1;
  int vertexActivation;
  Linear vertex2;
  Linear area1;
  int areaActivation;
  Linear area2;
  DeviceBuf globalToken;
  Linear global;

  vector<unique_ptr<TrunkBlock>> blocks;
  DeviceBuf tipGamma, tipBeta;
  float tipEpsilon;
  int tipActivation;

  Linear cand;
  Linear ground;
  Linear pool;
  vector<unique_ptr<Attention>> decoderAttentions;
  vector<unique_ptr<FFN>> decoderFFNs;
  DeviceBuf decoderNormWeight;
  float decoderNormEpsilon;
  Linear decoderOut;

  Linear v1;
  int v1Activation;
  Linear v2;
  int v2Activation;
  Linear v3;
  Linear sv3;
  Linear ownership;

  // Max widths over the layers, to size the work buffers
  int maxQDim = 0, maxKDim = 0, maxVDim = 0, maxAttDim = 0, maxFFN = 0, maxHP = 0;
  int maxDecQDim = 0, maxDecKDim = 0, maxDecVDim = 0, maxDecAttDim = 0, maxDecFFN = 0, maxDecHP = 0;

  // Work buffers
  DeviceBuf inputDev;
  DeviceBuf seqF, trunkH, tmp, mid, midN, q, k, v, att, ffA, ffG, delta, cosT, sinT, maskH, sumSq;
  DeviceBuf boundaryAtt, boundaryDelta;
  DeviceBuf pointEmb, vertexH1, vertexH, areaH1, areaH, globalProj;
  DeviceBuf pooled, poolProj, queries, queriesNorm, candEmb, groundEmb, dq, dk, dv, datt, dffA, dffG;
  DeviceBuf queriesF, cosQ, sinQ, policyH, policy;
  DeviceBuf outV1, pooledV, outV2, value, scoreValue, ownershipOut;

  Impl(const SparseDotsDesc& desc, bool fp16, bool mma, cudaStream_t stream_, cublasHandle_t cublas_, Logger* logger_)
    : useFP16(fp16), useMmaAttention(mma && fp16), stream(stream_), cublas(cublas_), logger(logger_) {
    areaRopeMean = desc.areaRopeMean;
    trunkChannels = desc.trunkNumChannels;
    midChannels = desc.midNumChannels;
    vertexChannels = desc.vertexMul1.outChannels;
    decoderChannels = desc.decoderNumChannels;
    testAssert(desc.numPointFeatures == SparseDotsNN::NUM_POINT_FEATURES);
    testAssert(desc.numVertexFeatures == SparseDotsNN::NUM_VERTEX_FEATURES);
    testAssert(desc.numCandFeatures == SparseDotsNN::NUM_CAND_FEATURES);
    testAssert(desc.numGlobalFeatures == NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE);
    testAssert(desc.areaMul1.inChannels == 2 * vertexChannels + SparseDotsNN::NUM_AREA_SCALAR_FEATURES);

    point.init(desc.pointMul, &desc.pointBias, false);
    vertex1.init(desc.vertexMul1, &desc.vertexBias1, false);
    vertexActivation = desc.vertexActivation.activation;
    vertex2.init(desc.vertexMul2, &desc.vertexBias2, false);
    area1.init(desc.areaMul1, &desc.areaBias1, false);
    areaActivation = desc.areaActivation.activation;
    area2.init(desc.areaMul2, &desc.areaBias2, false);
    testAssert(desc.globalToken.numChannels == trunkChannels);
    globalToken.upload(desc.globalToken.name, desc.globalToken.weights, false);
    global.init(desc.globalMul, nullptr, false);

    size_t numBoundaryOut = 0;
    for(const auto& blockPair : desc.blocks) {
      testAssert(blockPair.first == NESTED_BOTTLENECK_BLOCK_KIND);
      const auto& nested = *static_cast<const NestedBottleneckResidualBlockDesc*>(blockPair.second.get());
      auto block = std::make_unique<TrunkBlock>();
      block->preScale.upload(nested.preBN.name, nested.preBN.mergedScale, false);
      block->preBias.upload(nested.preBN.name, nested.preBN.mergedBias, false);
      block->preActivation = nested.preActivation.activation;
      block->preConv.initConv1x1(nested.preConv, useFP16);
      for(const auto& sub : nested.blocks) {
        if(sub.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
          auto attention = std::make_unique<Attention>();
          attention->init(*static_cast<const TransformerAttentionDesc*>(sub.second.get()), useFP16);
          maxQDim = std::max(maxQDim, attention->numHeads * attention->qHeadDim);
          maxKDim = std::max(maxKDim, attention->numKVHeads * attention->qHeadDim);
          maxVDim = std::max(maxVDim, attention->numKVHeads * attention->vHeadDim);
          maxAttDim = std::max(maxAttDim, attention->numHeads * attention->vHeadDim);
          maxHP = std::max(maxHP, attention->numKVHeadPairs());
          block->attentions.push_back(std::move(attention));
          block->isAttention.push_back(true);
          if(desc.areaBoundaryAttention) {
            auto boundaryOut = std::make_unique<Linear>();
            boundaryOut->init(desc.trunkBoundaryOut[numBoundaryOut++], nullptr, useFP16);
            block->boundaryOut.push_back(std::move(boundaryOut));
          }
        }
        else {
          testAssert(sub.first == TRANSFORMER_FFN_BLOCK_KIND);
          auto ffn = std::make_unique<FFN>();
          ffn->init(*static_cast<const TransformerFFNDesc*>(sub.second.get()), useFP16);
          maxFFN = std::max(maxFFN, ffn->linear1.outC);
          block->ffns.push_back(std::move(ffn));
          block->isAttention.push_back(false);
        }
      }
      block->postScale.upload(nested.postBN.name, nested.postBN.mergedScale, false);
      block->postBias.upload(nested.postBN.name, nested.postBN.mergedBias, false);
      block->postActivation = nested.postActivation.activation;
      block->postConv.initConv1x1(nested.postConv, useFP16);
      testAssert(block->preConv.outC == midChannels && block->postConv.outC == trunkChannels);
      blocks.push_back(std::move(block));
    }
    testAssert(numBoundaryOut == desc.trunkBoundaryOut.size());
    tipGamma.upload(desc.trunkTipRMSNorm.name, desc.trunkTipRMSNorm.gamma, false);
    tipBeta.upload(desc.trunkTipRMSNorm.name, desc.trunkTipRMSNorm.beta, false);
    tipEpsilon = desc.trunkTipRMSNorm.epsilon;
    tipActivation = desc.trunkTipActivation.activation;

    cand.init(desc.candMul, &desc.candBias, false);
    ground.init(desc.groundMul, &desc.groundBias, false);
    pool.init(desc.poolMul, nullptr, false);
    testAssert(desc.decoderAttention.size() == desc.decoderFFN.size());
    for(size_t l = 0; l < desc.decoderAttention.size(); l++) {
      auto attention = std::make_unique<Attention>();
      attention->init(desc.decoderAttention[l], useFP16);
      maxDecQDim = std::max(maxDecQDim, attention->numHeads * attention->qHeadDim);
      maxDecKDim = std::max(maxDecKDim, attention->numKVHeads * attention->qHeadDim);
      maxDecVDim = std::max(maxDecVDim, attention->numKVHeads * attention->vHeadDim);
      maxDecAttDim = std::max(maxDecAttDim, attention->numHeads * attention->vHeadDim);
      maxDecHP = std::max(maxDecHP, attention->numKVHeadPairs());
      decoderAttentions.push_back(std::move(attention));
      auto ffn = std::make_unique<FFN>();
      ffn->init(desc.decoderFFN[l], useFP16);
      maxDecFFN = std::max(maxDecFFN, ffn->linear1.outC);
      decoderFFNs.push_back(std::move(ffn));
    }
    decoderNormWeight.upload(desc.decoderFinalNorm.name, desc.decoderFinalNorm.weight, false);
    decoderNormEpsilon = desc.decoderFinalNorm.epsilon;
    decoderOut.init(desc.decoderOutMul, nullptr, useFP16);
    testAssert(decoderOut.outC >= 2);

    v1.init(desc.v1Mul, &desc.v1Bias, false);
    v1Activation = desc.v1Activation.activation;
    v2.init(desc.v2Mul, &desc.v2Bias, false);
    v2Activation = desc.v2Activation.activation;
    v3.init(desc.v3Mul, &desc.v3Bias, false);
    sv3.init(desc.sv3Mul, &desc.sv3Bias, false);
    ownership.init(desc.ownershipMul, nullptr, false);
    testAssert(v3.outC == 3);
    testAssert(sv3.outC == 6);
    testAssert(ownership.outC == 1);
  }

  // out[rows x outC] (+)= in[rows x inC] * w, then bias and activation
  void gemm(const Linear& layer, const void* in, long long rows, void* out, bool accumulate = false,
            int activation = ACTIVATION_IDENTITY) {
    if(rows <= 0)
      return;
    testAssert(rows <= INT_MAX);
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    const cudaDataType_t type = layer.useFP16 ? CUDA_R_16F : CUDA_R_32F;
    CUBLAS_ERR(layer.name.c_str(), cublasGemmEx(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N,
      layer.outC, (int)rows, layer.inC,
      &alpha,
      layer.w.ptr, type, layer.outC,
      in, type, layer.inC,
      &beta,
      out, type, layer.outC,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    if(layer.useFP16)
      sparseDotsBiasAct<half>((half*)out, layer.b.f(), rows, layer.outC, activation, stream);
    else
      sparseDotsBiasAct<float>((float*)out, layer.b.f(), rows, layer.outC, activation, stream);
  }

  // Positions of the batch (fixed for one evaluate call)
  struct Batch {
    int n = 0;    // number of positions
    int s = 0;    // padded trunk tokens per position
    int qLen = 0; // padded candidate queries per position
    int totalPoints = 0, totalAreas = 0, totalVertices = 0, totalCands = 0, totalGrounds = 0;
    int totalTokPos = 0, totalCandPos = 0;
    // Device pointers into inputDev
    float* pointFeat; int* pointDst;
    float* vertexFeat; int* vertexSeg; int* vertexPointRow;
    float* areaAgg; int* areaDst;
    float* global;
    float* mask; float* numTokens; float* boardOffset;
    int* tokSeg; float* tokPos;
    float* candFeat; int* candDst;
    float* groundFeat; int* groundDst;
    int* candSeg; float* candPos;
  };

  void pack(const vector<const DotsSparseInput*>& inputs, Batch& b, HostPack& host) {
    b.n = (int)inputs.size();
    int maxTokens = 1;
    int maxQueries = 1;
    for(const DotsSparseInput* in : inputs) {
      const int numPoints = (int)in->points.size();
      const int numAreas = (int)in->areas.size();
      maxTokens = std::max(maxTokens, 1 + numPoints + numAreas);
      maxQueries = std::max(maxQueries, (int)in->candidates.size() + (in->groundAllowed ? 1 : 0));
      b.totalPoints += numPoints;
      b.totalAreas += numAreas;
      b.totalVertices += (int)in->vertices.size();
      b.totalCands += (int)in->candidates.size();
      b.totalGrounds += in->groundAllowed ? 1 : 0;
      b.totalTokPos += numPoints + (areaRopeMean ? (int)in->vertices.size() : 0);
      b.totalCandPos += (int)in->candidates.size();
    }
    b.s = (maxTokens + SEQ_LEN_ALIGN - 1) / SEQ_LEN_ALIGN * SEQ_LEN_ALIGN;
    b.qLen = maxQueries;
    const int numTokens = b.n * b.s;
    const int numQueries = b.n * b.qLen;
    const int aggC = 2 * vertexChannels + SparseDotsNN::NUM_AREA_SCALAR_FEATURES;

    host.size = 0;
    const size_t oPointFeat = host.reserve(sizeof(float) * b.totalPoints * SparseDotsNN::NUM_POINT_FEATURES);
    const size_t oPointDst = host.reserve(sizeof(int) * b.totalPoints);
    const size_t oVertexFeat = host.reserve(sizeof(float) * b.totalVertices * SparseDotsNN::NUM_VERTEX_FEATURES);
    const size_t oVertexSeg = host.reserve(sizeof(int) * (b.totalAreas + 1));
    const size_t oVertexPointRow = host.reserve(sizeof(int) * b.totalVertices);
    const size_t oAreaAgg = host.reserve(sizeof(float) * (size_t)b.totalAreas * aggC);
    const size_t oAreaDst = host.reserve(sizeof(int) * b.totalAreas);
    const size_t oGlobal = host.reserve(sizeof(float) * b.n * NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE);
    const size_t oMask = host.reserve(sizeof(float) * numTokens);
    const size_t oNumTokens = host.reserve(sizeof(float) * b.n);
    const size_t oBoardOffset = host.reserve(sizeof(float) * b.n);
    const size_t oTokSeg = host.reserve(sizeof(int) * (numTokens + 1));
    const size_t oTokPos = host.reserve(sizeof(float) * 3 * b.totalTokPos);
    const size_t oCandFeat = host.reserve(sizeof(float) * b.totalCands * SparseDotsNN::NUM_CAND_FEATURES);
    const size_t oCandDst = host.reserve(sizeof(int) * b.totalCands);
    const size_t oGroundFeat = host.reserve(sizeof(float) * b.totalGrounds);
    const size_t oGroundDst = host.reserve(sizeof(int) * b.totalGrounds);
    const size_t oCandSeg = host.reserve(sizeof(int) * (numQueries + 1));
    const size_t oCandPos = host.reserve(sizeof(float) * 3 * b.totalCandPos);
    host.data.assign(host.size, 0);

    char* h = host.data.data();
    float* pointFeat = (float*)(h + oPointFeat);
    int* pointDst = (int*)(h + oPointDst);
    float* vertexFeat = (float*)(h + oVertexFeat);
    int* vertexSeg = (int*)(h + oVertexSeg);
    int* vertexPointRow = (int*)(h + oVertexPointRow);
    float* areaAgg = (float*)(h + oAreaAgg);
    int* areaDst = (int*)(h + oAreaDst);
    float* global = (float*)(h + oGlobal);
    float* mask = (float*)(h + oMask);
    float* numTokensHost = (float*)(h + oNumTokens);
    float* boardOffset = (float*)(h + oBoardOffset);
    int* tokSeg = (int*)(h + oTokSeg);
    float* tokPos = (float*)(h + oTokPos);
    float* candFeat = (float*)(h + oCandFeat);
    int* candDst = (int*)(h + oCandDst);
    float* groundFeat = (float*)(h + oGroundFeat);
    int* groundDst = (int*)(h + oGroundDst);
    int* candSeg = (int*)(h + oCandSeg);
    float* candPos = (float*)(h + oCandPos);

    int pointRow = 0, vertexRow = 0, areaRow = 0, candRow = 0, groundRow = 0, tokPosRow = 0, candPosRow = 0;
    for(int n = 0; n < b.n; n++) {
      const DotsSparseInput& in = *inputs[n];
      const int numPoints = (int)in.points.size();
      const int numAreas = (int)in.areas.size();
      const int numCands = (int)in.candidates.size();
      const int tokenBase = n * b.s;
      std::copy(in.global, in.global + NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE,
                global + (size_t)n * NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE);
      for(int t = 0; t < 1 + numPoints + numAreas; t++)
        mask[tokenBase + t] = 1.0f;
      numTokensHost[n] = (float)(1 + numPoints + numAreas);
      boardOffset[n] = sqrtf((float)in.xSize * (float)in.ySize) - 14.0f;

      for(int i = 0; i < numPoints; i++) {
        SparseDotsNN::pointFeatures(in, i, pointFeat + (size_t)(pointRow + i) * SparseDotsNN::NUM_POINT_FEATURES);
        pointDst[pointRow + i] = tokenBase + 1 + i;
      }
      const int vertexBase = vertexRow;
      testAssert(in.vertexPoints.size() == in.vertices.size());
      for(int i = 0; i < (int)in.vertices.size(); i++) {
        SparseDotsNN::vertexFeatures(in, i, vertexFeat + (size_t)(vertexRow + i) * SparseDotsNN::NUM_VERTEX_FEATURES);
        vertexPointRow[vertexRow + i] = tokenBase + 1 + in.vertexPoints[i];
      }
      int expectedStart = 0;
      for(int a = 0; a < numAreas; a++) {
        const auto& area = in.areas[a];
        testAssert(area.vertexStart == expectedStart);
        expectedStart += area.boundaryLen;
        vertexSeg[areaRow + a] = vertexBase + area.vertexStart;
        SparseDotsNN::areaScalarFeatures(in, a, areaAgg + (size_t)(areaRow + a) * aggC + 2 * vertexChannels);
        areaDst[areaRow + a] = tokenBase + 1 + numPoints + a;
      }
      testAssert(expectedStart == (int)in.vertices.size());

      // RoPE positions of the trunk tokens: GLOBAL and padding have none
      tokSeg[tokenBase] = tokPosRow;
      for(int t = 1; t < b.s; t++) {
        tokSeg[tokenBase + t] = tokPosRow;
        if(t <= numPoints) {
          const auto& p = in.points[t - 1];
          tokPos[3 * tokPosRow] = p.x;
          tokPos[3 * tokPosRow + 1] = p.y;
          tokPos[3 * tokPosRow + 2] = 1.0f;
          tokPosRow++;
        }
        else if(t <= numPoints + numAreas && areaRopeMean) {
          const auto& area = in.areas[t - 1 - numPoints];
          const float weight = 1.0f / (float)std::max<int>(1, area.boundaryLen);
          for(int vtx = 0; vtx < area.boundaryLen; vtx++) {
            const auto& vertex = in.vertices[area.vertexStart + vtx];
            tokPos[3 * tokPosRow] = vertex.x;
            tokPos[3 * tokPosRow + 1] = vertex.y;
            tokPos[3 * tokPosRow + 2] = weight;
            tokPosRow++;
          }
        }
      }

      // Candidate queries: point candidates then grounding (no position), then padding
      const int queryBase = n * b.qLen;
      for(int i = 0; i < numCands; i++) {
        SparseDotsNN::candidateFeatures(in, i, candFeat + (size_t)(candRow + i) * SparseDotsNN::NUM_CAND_FEATURES);
        candDst[candRow + i] = queryBase + i;
      }
      if(in.groundAllowed) {
        groundFeat[groundRow] = SparseDotsNN::groundFeature(in);
        groundDst[groundRow] = queryBase + numCands;
        groundRow++;
      }
      for(int i = 0; i < b.qLen; i++) {
        candSeg[queryBase + i] = candPosRow;
        if(i < numCands) {
          candPos[3 * candPosRow] = in.candidates[i].x;
          candPos[3 * candPosRow + 1] = in.candidates[i].y;
          candPos[3 * candPosRow + 2] = 1.0f;
          candPosRow++;
        }
      }

      pointRow += numPoints;
      vertexRow += (int)in.vertices.size();
      areaRow += numAreas;
      candRow += numCands;
    }
    vertexSeg[b.totalAreas] = b.totalVertices;
    tokSeg[numTokens] = tokPosRow;
    candSeg[numQueries] = candPosRow;
    testAssert(tokPosRow == b.totalTokPos && candPosRow == b.totalCandPos && groundRow == b.totalGrounds);

    inputDev.ensure(host.size);
    char* d = (char*)inputDev.ptr;
    CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(d, h, host.size, cudaMemcpyHostToDevice, stream));
    b.pointFeat = (float*)(d + oPointFeat);
    b.pointDst = (int*)(d + oPointDst);
    b.vertexFeat = (float*)(d + oVertexFeat);
    b.vertexSeg = (int*)(d + oVertexSeg);
    b.vertexPointRow = (int*)(d + oVertexPointRow);
    b.areaAgg = (float*)(d + oAreaAgg);
    b.areaDst = (int*)(d + oAreaDst);
    b.global = (float*)(d + oGlobal);
    b.mask = (float*)(d + oMask);
    b.numTokens = (float*)(d + oNumTokens);
    b.boardOffset = (float*)(d + oBoardOffset);
    b.tokSeg = (int*)(d + oTokSeg);
    b.tokPos = (float*)(d + oTokPos);
    b.candFeat = (float*)(d + oCandFeat);
    b.candDst = (int*)(d + oCandDst);
    b.groundFeat = (float*)(d + oGroundFeat);
    b.groundDst = (int*)(d + oGroundDst);
    b.candSeg = (int*)(d + oCandSeg);
    b.candPos = (float*)(d + oCandPos);
  }

  void ensureBuffers(const Batch& b) {
    const size_t tokens = (size_t)b.n * b.s;
    const size_t queries = (size_t)b.n * b.qLen;
    const size_t actBytes = useFP16 ? sizeof(half) : sizeof(float);
    const size_t fb = sizeof(float);
    const int c = trunkChannels;
    const int m = midChannels;
    const int cd = decoderChannels;

    seqF.ensure(tokens * c * fb);
    if(useFP16) {
      trunkH.ensure(tokens * c * actBytes);
      maskH.ensure(tokens * sizeof(half));
    }
    tmp.ensure(tokens * c * actBytes);
    mid.ensure(tokens * m * actBytes);
    midN.ensure(tokens * m * actBytes);
    q.ensure(tokens * maxQDim * actBytes);
    k.ensure(tokens * maxKDim * actBytes);
    v.ensure(tokens * maxVDim * actBytes);
    att.ensure(tokens * maxAttDim * actBytes);
    ffA.ensure(tokens * maxFFN * actBytes);
    ffG.ensure(tokens * maxFFN * actBytes);
    delta.ensure(tokens * m * actBytes);
    boundaryAtt.ensure((size_t)b.totalAreas * maxAttDim * actBytes);
    boundaryDelta.ensure((size_t)b.totalAreas * m * actBytes);
    cosT.ensure(tokens * std::max(maxHP, maxDecHP) * fb);
    sinT.ensure(tokens * std::max(maxHP, maxDecHP) * fb);
    sumSq.ensure(b.n * fb);

    pointEmb.ensure((size_t)b.totalPoints * c * fb);
    vertexH1.ensure((size_t)b.totalVertices * vertexChannels * fb);
    vertexH.ensure((size_t)b.totalVertices * vertexChannels * fb);
    areaH1.ensure((size_t)b.totalAreas * area1.outC * fb);
    areaH.ensure((size_t)b.totalAreas * c * fb);
    globalProj.ensure((size_t)b.n * c * fb);

    pooled.ensure((size_t)b.n * 3 * c * fb);
    poolProj.ensure((size_t)b.n * cd * fb);
    this->queries.ensure(queries * cd * actBytes);
    queriesNorm.ensure(queries * cd * actBytes);
    if(useFP16) {
      queriesF.ensure(queries * cd * fb);
      policyH.ensure(queries * decoderOut.outC * actBytes);
    }
    candEmb.ensure((size_t)b.totalCands * cd * fb);
    groundEmb.ensure((size_t)b.totalGrounds * cd * fb);
    dq.ensure(queries * maxDecQDim * actBytes);
    dk.ensure(tokens * maxDecKDim * actBytes);
    dv.ensure(tokens * maxDecVDim * actBytes);
    datt.ensure(queries * maxDecAttDim * actBytes);
    dffA.ensure(queries * maxDecFFN * actBytes);
    dffG.ensure(queries * maxDecFFN * actBytes);
    cosQ.ensure(queries * maxDecHP * fb);
    sinQ.ensure(queries * maxDecHP * fb);
    policy.ensure(queries * decoderOut.outC * fb);

    outV1.ensure(tokens * v1.outC * fb);
    pooledV.ensure((size_t)b.n * 3 * v1.outC * fb);
    outV2.ensure((size_t)b.n * v2.outC * fb);
    value.ensure((size_t)b.n * v3.outC * fb);
    scoreValue.ensure((size_t)b.n * sv3.outC * fb);
    ownershipOut.ensure(tokens * fb);
  }

  template <typename T>
  void trunkAttention(const Attention& layer, const Linear* boundaryOut, const Batch& b) {
    const long long rows = (long long)b.n * b.s;
    T* midT = (T*)mid.ptr;
    T* midNT = (T*)midN.ptr;
    T* qT = (T*)q.ptr;
    T* kT = (T*)k.ptr;
    T* vT = (T*)v.ptr;
    T* attT = (T*)att.ptr;
    sparseDotsRMSNorm<T>(midT, midNT, layer.normWeight.f(), b.mask, rows, midChannels, layer.epsilon, stream);
    gemm(layer.q, midNT, rows, qT);
    gemm(layer.k, midNT, rows, kT);
    gemm(layer.v, midNT, rows, vT);
    sparseDotsRopeTables(layer.ropeFreqs.f(), b.tokSeg, b.tokPos, (int)rows, layer.numKVHeadPairs(), cosT.f(), sinT.f(), stream);
    sparseDotsApplyRope<T>(qT, cosT.f(), sinT.f(), (int)rows, layer.numHeads, layer.numKVHeads, layer.qHeadDim,
                           layer.numHeads * layer.qHeadDim, stream);
    sparseDotsApplyRope<T>(kT, cosT.f(), sinT.f(), (int)rows, layer.numKVHeads, layer.numKVHeads, layer.qHeadDim,
                           layer.numKVHeads * layer.qHeadDim, stream);
    bool done = false;
    if constexpr(std::is_same<T, half>::value) {
      if(useMmaAttention && layer.qHeadDim == layer.vHeadDim) {
        done = customCudaFlashAttentionMma(
          qT, kT, vT, (const half*)maskH.ptr, attT, b.n, b.s, layer.numHeads, layer.numKVHeads, layer.qHeadDim,
          layer.vHeadDim, layer.numHeads * layer.qHeadDim, layer.numKVHeads * layer.qHeadDim, stream);
        if(done && !loggedMma) {
          loggedMma = true;
          if(logger != nullptr)
            logger->write("Sparse Dots CUDA: using tensor-core mma flash attention kernel");
        }
      }
    }
    if(!done)
      sparseDotsAttention<T>(qT, kT, vT, b.mask, attT, b.n, b.s, b.s, layer.numHeads, layer.numKVHeads, layer.qHeadDim,
                             layer.vHeadDim, stream);
    gemm(layer.out, attT, rows, delta.ptr);
    sparseDotsResidualAdd<T>(midT, (const T*)delta.ptr, b.mask, rows, midChannels, stream);

    // The areas also attend to the points of their own boundary only, with the same q, k, v
    if(boundaryOut != nullptr && b.totalAreas > 0) {
      sparseDotsBoundaryAttention<T>(
        qT, kT, vT, b.areaDst, b.vertexSeg, b.vertexPointRow, (T*)boundaryAtt.ptr, b.totalAreas, layer.numHeads,
        layer.numKVHeads, layer.qHeadDim, layer.vHeadDim, stream);
      gemm(*boundaryOut, boundaryAtt.ptr, b.totalAreas, boundaryDelta.ptr);
      sparseDotsScatterAddRows<T>((const T*)boundaryDelta.ptr, midT, b.areaDst, b.totalAreas, midChannels, stream);
    }
  }

  template <typename T>
  void ffnResidual(const FFN& ffn, T* x, T* xNorm, T* a, T* g, const float* mask, long long rows) {
    sparseDotsRMSNorm<T>(x, xNorm, ffn.normWeight.f(), mask, rows, ffn.linear1.inC, ffn.epsilon, stream);
    gemm(ffn.linear1, xNorm, rows, a);
    gemm(ffn.gate, xNorm, rows, g);
    sparseDotsSwiGLU<T>(a, g, rows * ffn.linear1.outC, stream);
    if(mask == nullptr) {
      gemm(ffn.linear2, a, rows, x, true);
    }
    else {
      gemm(ffn.linear2, a, rows, delta.ptr);
      sparseDotsResidualAdd<T>(x, (const T*)delta.ptr, mask, rows, ffn.linear2.outC, stream);
    }
  }

  // Runs the trunk on trunkBuf in place, ends with the tip norm
  template <typename T>
  void runTrunk(T* trunk, const Batch& b) {
    const long long rows = (long long)b.n * b.s;
    T* tmpT = (T*)tmp.ptr;
    T* midT = (T*)mid.ptr;
    T* midNT = (T*)midN.ptr;
    for(const auto& block : blocks) {
      sparseDotsScaleBiasAct<T>(trunk, tmpT, block->preScale.f(), block->preBias.f(), b.mask, rows, trunkChannels,
                                block->preActivation, stream);
      gemm(block->preConv, tmpT, rows, midT);
      size_t attentionIdx = 0;
      size_t ffnIdx = 0;
      for(bool isAttention : block->isAttention) {
        if(isAttention) {
          const Linear* boundaryOut = block->boundaryOut.empty() ? nullptr : block->boundaryOut[attentionIdx].get();
          trunkAttention<T>(*block->attentions[attentionIdx++], boundaryOut, b);
        }
        else
          ffnResidual<T>(*block->ffns[ffnIdx++], midT, midNT, (T*)ffA.ptr, (T*)ffG.ptr, b.mask, rows);
      }
      sparseDotsScaleBiasAct<T>(midT, midNT, block->postScale.f(), block->postBias.f(), b.mask, rows, midChannels,
                                block->postActivation, stream);
      // Masked rows of midN are zero and the conv has no bias, so the residual add folds into the GEMM
      gemm(block->postConv, midNT, rows, trunk, true);
    }
    sparseDotsSpatialRMSNorm<T>(trunk, trunk, tipGamma.f(), tipBeta.f(), b.mask, b.numTokens, b.n, b.s, trunkChannels,
                                tipEpsilon, tipActivation, sumSq.f(), stream);
  }

  // Runs the decoder layers on the embedded queries, writes the logits to policy (FP32) or policyH (FP16)
  template <typename T>
  void runDecoder(const T* trunk, const Batch& b) {
    const long long tokens = (long long)b.n * b.s;
    const long long numQueries = (long long)b.n * b.qLen;
    const int cd = decoderChannels;
    T* query = (T*)queries.ptr;
    T* queryNorm = (T*)queriesNorm.ptr;
    T* dqT = (T*)dq.ptr;
    T* dkT = (T*)dk.ptr;
    T* dvT = (T*)dv.ptr;
    T* dattT = (T*)datt.ptr;
    for(size_t l = 0; l < decoderAttentions.size(); l++) {
      const Attention& layer = *decoderAttentions[l];
      sparseDotsRMSNorm<T>(query, queryNorm, layer.normWeight.f(), nullptr, numQueries, cd, layer.epsilon, stream);
      gemm(layer.q, queryNorm, numQueries, dqT);
      gemm(layer.k, trunk, tokens, dkT);
      gemm(layer.v, trunk, tokens, dvT);
      const int hp = layer.numKVHeadPairs();
      sparseDotsRopeTables(layer.ropeFreqs.f(), b.candSeg, b.candPos, (int)numQueries, hp, cosQ.f(), sinQ.f(), stream);
      sparseDotsRopeTables(layer.ropeFreqs.f(), b.tokSeg, b.tokPos, (int)tokens, hp, cosT.f(), sinT.f(), stream);
      sparseDotsApplyRope<T>(dqT, cosQ.f(), sinQ.f(), (int)numQueries, layer.numHeads, layer.numKVHeads, layer.qHeadDim,
                             layer.numHeads * layer.qHeadDim, stream);
      sparseDotsApplyRope<T>(dkT, cosT.f(), sinT.f(), (int)tokens, layer.numKVHeads, layer.numKVHeads, layer.qHeadDim,
                             layer.numKVHeads * layer.qHeadDim, stream);
      bool done = false;
      if constexpr(std::is_same<T, half>::value) {
        if(useMmaAttention && layer.qHeadDim == layer.vHeadDim)
          done = customCudaFlashAttentionMmaCross(
            dqT, dkT, dvT, (const half*)maskH.ptr, dattT, b.n, b.qLen, b.s, layer.numHeads, layer.numKVHeads,
            layer.qHeadDim, layer.vHeadDim, layer.numHeads * layer.qHeadDim, layer.numKVHeads * layer.qHeadDim, stream);
      }
      if(!done)
        sparseDotsAttention<T>(dqT, dkT, dvT, b.mask, dattT, b.n, b.qLen, b.s, layer.numHeads, layer.numKVHeads,
                               layer.qHeadDim, layer.vHeadDim, stream);
      gemm(layer.out, dattT, numQueries, query, true);
      ffnResidual<T>(*decoderFFNs[l], query, queryNorm, (T*)dffA.ptr, (T*)dffG.ptr, nullptr, numQueries);
    }
    sparseDotsRMSNorm<T>(query, queryNorm, decoderNormWeight.f(), nullptr, numQueries, cd, decoderNormEpsilon, stream);
    gemm(decoderOut, queryNorm, numQueries, std::is_same<T, half>::value ? policyH.ptr : policy.ptr);
  }

  void evaluate(const vector<const DotsSparseInput*>& inputs, vector<SparseDotsNN::RawOutput>& outputs) {
    testAssert(!inputs.empty());
    Batch b;
    HostPack host;
    const auto hostStart = std::chrono::steady_clock::now();
    pack(inputs, b, host);
    ensureBuffers(b);
    const double hostMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hostStart).count();
    mark("start");
    const long long tokens = (long long)b.n * b.s;
    const long long queries = (long long)b.n * b.qLen;
    const int c = trunkChannels;
    const int cd = decoderChannels;

    // Embeddings in FP32 --------------------------------------------------------------------------------------------
    float* seq = seqF.f();
    CUDA_ERR("SparseDotsCudaModel", cudaMemsetAsync(seq, 0, sizeof(float) * tokens * c, stream));
    gemm(point, b.pointFeat, b.totalPoints, pointEmb.ptr);
    sparseDotsScatterRows(pointEmb.f(), seq, b.pointDst, b.totalPoints, c, stream);
    if(b.totalAreas > 0) {
      const int aggC = 2 * vertexChannels + SparseDotsNN::NUM_AREA_SCALAR_FEATURES;
      gemm(vertex1, b.vertexFeat, b.totalVertices, vertexH1.ptr, false, vertexActivation);
      gemm(vertex2, vertexH1.ptr, b.totalVertices, vertexH.ptr);
      sparseDotsSegmentMeanMax(vertexH.f(), b.vertexSeg, b.totalAreas, vertexChannels, b.areaAgg, aggC, stream);
      gemm(area1, b.areaAgg, b.totalAreas, areaH1.ptr, false, areaActivation);
      gemm(area2, areaH1.ptr, b.totalAreas, areaH.ptr);
      sparseDotsScatterRows(areaH.f(), seq, b.areaDst, b.totalAreas, c, stream);
    }
    gemm(global, b.global, b.n, globalProj.ptr);
    sparseDotsAddGlobal(seq, globalToken.f(), globalProj.f(), b.mask, b.n, b.s, c, stream);

    mark("embeddings");

    // Trunk -----------------------------------------------------------------------------------------------------------
    if(useFP16) {
      sparseDotsCopyToHalf(b.mask, (half*)maskH.ptr, tokens, stream);
      sparseDotsCopyToHalf(seq, (half*)trunkH.ptr, tokens * c, stream);
      runTrunk<half>((half*)trunkH.ptr, b);
      sparseDotsCopyFromHalf((const half*)trunkH.ptr, seq, tokens * c, stream);
    }
    else {
      runTrunk<float>(seq, b);
    }
    const float* trunk = seq;
    mark("trunk");

    // Candidate decoder ---------------------------------------------------------------------------------------------
    sparseDotsPool(trunk, b.mask, b.numTokens, b.boardOffset, pooled.f(), b.n, b.s, c, false, stream);
    gemm(pool, pooled.ptr, b.n, poolProj.ptr);
    float* queryF = useFP16 ? queriesF.f() : (float*)this->queries.ptr;
    CUDA_ERR("SparseDotsCudaModel", cudaMemsetAsync(queryF, 0, sizeof(float) * queries * cd, stream));
    gemm(cand, b.candFeat, b.totalCands, candEmb.ptr);
    sparseDotsScatterRows(candEmb.f(), queryF, b.candDst, b.totalCands, cd, stream);
    gemm(ground, b.groundFeat, b.totalGrounds, groundEmb.ptr);
    sparseDotsScatterRows(groundEmb.f(), queryF, b.groundDst, b.totalGrounds, cd, stream);
    sparseDotsAddRowVector(queryF, poolProj.f(), b.n, b.qLen, cd, stream);
    if(useFP16) {
      sparseDotsCopyToHalf(queryF, (half*)this->queries.ptr, queries * cd, stream);
      runDecoder<half>((const half*)trunkH.ptr, b);
      sparseDotsCopyFromHalf((const half*)policyH.ptr, policy.f(), queries * decoderOut.outC, stream);
    }
    else {
      runDecoder<float>(trunk, b);
    }
    mark("decoder");

    // Value head in FP32 ----------------------------------------------------------------------------------------------
    gemm(v1, trunk, tokens, outV1.ptr, false, v1Activation);
    sparseDotsPool(outV1.f(), b.mask, b.numTokens, b.boardOffset, pooledV.f(), b.n, b.s, v1.outC, true, stream);
    gemm(v2, pooledV.ptr, b.n, outV2.ptr, false, v2Activation);
    gemm(v3, outV2.ptr, b.n, value.ptr);
    gemm(sv3, outV2.ptr, b.n, scoreValue.ptr);
    gemm(ownership, outV1.ptr, tokens, ownershipOut.ptr);
    CUDA_ERR("SparseDotsCudaModel", cudaPeekAtLastError());
    mark("value head");

    // Outputs ---------------------------------------------------------------------------------------------------------
    const int policyC = decoderOut.outC;
    vector<float> policyHost((size_t)queries * policyC);
    vector<float> valueHost((size_t)b.n * 3);
    vector<float> scoreHost((size_t)b.n * 6);
    vector<float> ownershipHost((size_t)tokens);
    CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(policyHost.data(), policy.ptr, sizeof(float) * policyHost.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(valueHost.data(), value.ptr, sizeof(float) * valueHost.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(scoreHost.data(), scoreValue.ptr, sizeof(float) * scoreHost.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(ownershipHost.data(), ownershipOut.ptr, sizeof(float) * ownershipHost.size(), cudaMemcpyDeviceToHost, stream));
    mark("download");
    CUDA_ERR("SparseDotsCudaModel", cudaStreamSynchronize(stream));
    finishProfile(hostMs, b.n);

    outputs.resize(b.n);
    for(int n = 0; n < b.n; n++) {
      const DotsSparseInput& in = *inputs[n];
      SparseDotsNN::RawOutput& out = outputs[n];
      out.numCandidates = (int)in.candidates.size();
      out.hasGround = in.groundAllowed;
      const int numQueries = out.numCandidates + (out.hasGround ? 1 : 0);
      out.policy.resize((size_t)numQueries * 2);
      for(int i = 0; i < numQueries; i++) {
        out.policy[(size_t)i * 2] = policyHost[((size_t)n * b.qLen + i) * policyC];
        out.policy[(size_t)i * 2 + 1] = policyHost[((size_t)n * b.qLen + i) * policyC + 1];
      }
      for(int i = 0; i < 3; i++)
        out.value[i] = valueHost[(size_t)n * 3 + i];
      for(int i = 0; i < 6; i++)
        out.scoreValue[i] = scoreHost[(size_t)n * 6 + i];
      out.ownership.resize(in.points.size());
      for(size_t i = 0; i < in.points.size(); i++)
        out.ownership[i] = ownershipHost[(size_t)n * b.s + 1 + i];
    }
  }
};

SparseDotsCudaModel::SparseDotsCudaModel(
  const SparseDotsDesc& desc, bool useFP16, bool useMmaAttention, cudaStream_t stream, cublasHandle_t cublas, Logger* logger)
  : impl(std::make_unique<Impl>(desc, useFP16, useMmaAttention, stream, cublas, logger)) {}

SparseDotsCudaModel::~SparseDotsCudaModel() = default;

void SparseDotsCudaModel::evaluate(const vector<const DotsSparseInput*>& inputs, vector<SparseDotsNN::RawOutput>& outputs) {
  impl->evaluate(inputs, outputs);
}
