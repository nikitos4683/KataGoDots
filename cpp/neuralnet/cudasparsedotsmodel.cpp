#include "../neuralnet/cudasparsedotsmodel.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

  // Grows the buffer to at least the given size, dropping the content. Grows with a margin, the batches vary.
  void ensure(size_t need) {
    if(need <= bytes)
      return;
    const size_t newBytes = std::max(need, bytes + bytes / 4);
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

// A pinned host buffer that only grows, for asynchronous copies
struct PinnedBuf {
  char* ptr = nullptr;
  size_t bytes = 0;

  PinnedBuf() {}
  ~PinnedBuf() {
    if(ptr != nullptr)
      (void)cudaFreeHost(ptr);
  }
  PinnedBuf(const PinnedBuf&) = delete;
  PinnedBuf& operator=(const PinnedBuf&) = delete;

  void ensure(size_t need) {
    if(need <= bytes)
      return;
    const size_t newBytes = std::max(need, bytes + bytes / 4);
    if(ptr != nullptr)
      (void)cudaFreeHost(ptr);
    ptr = nullptr;
    bytes = 0;
    CUDA_ERR("SparseDotsCudaModel", cudaHostAlloc((void**)&ptr, newBytes, cudaHostAllocDefault));
    bytes = newBytes;
  }
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
  // Several layers with the same input as one: their outputs are concatenated in each row
  void initConcat(const vector<const MatMulLayerDesc*>& descs, bool fp16) {
    testAssert(!descs.empty());
    name = descs[0]->name;
    inC = descs[0]->inChannels;
    outC = 0;
    useFP16 = fp16;
    for(const MatMulLayerDesc* desc : descs) {
      testAssert(desc->inChannels == inC);
      testAssert(desc->weights.size() == (size_t)inC * desc->outChannels);
      outC += desc->outChannels;
    }
    vector<float> weights((size_t)inC * outC);
    int offset = 0;
    for(const MatMulLayerDesc* desc : descs) {
      const int oc = desc->outChannels;
      for(int ic = 0; ic < inC; ic++)
        for(int o = 0; o < oc; o++)
          weights[(size_t)ic * outC + offset + o] = desc->weights[(size_t)ic * oc + o];
      offset += oc;
    }
    w.upload(name, weights, useFP16);
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

// Self-attention (selfAttention) projects q, k and v of the same rows with one GEMM into rows [q | k | v].
// Cross-attention projects q of the queries, and k and v of the keys with one GEMM into rows [k | v].
struct Attention {
  int numHeads = 0;
  int numKVHeads = 0;
  int qHeadDim = 0;
  int vHeadDim = 0;
  float epsilon = 0.0f;
  DeviceBuf normWeight;
  Linear qkv;
  Linear q, kv;
  Linear out;
  DeviceBuf ropeFreqs;

  void init(const TransformerAttentionDesc& desc, bool selfAttention, bool fp16) {
    testAssert(desc.useRope && desc.learnableRope);
    numHeads = desc.numHeads;
    numKVHeads = desc.numKVHeads;
    qHeadDim = desc.qHeadDim;
    vHeadDim = desc.vHeadDim;
    epsilon = desc.preLN.epsilon;
    testAssert(qHeadDim % 2 == 0);
    testAssert(desc.ropeFreqs.size() == (size_t)numKVHeads * (qHeadDim / 2) * 2);
    normWeight.upload(desc.preLN.name, desc.preLN.weight, false);
    if(selfAttention) {
      qkv.initConcat({&desc.qProj, &desc.kProj, &desc.vProj}, fp16);
    }
    else {
      q.init(desc.qProj, nullptr, fp16);
      kv.initConcat({&desc.kProj, &desc.vProj}, fp16);
    }
    out.init(desc.outProj, nullptr, fp16);
    ropeFreqs.upload(desc.name + "/ropeFreqs", desc.ropeFreqs, false);
  }
  int qDim() const { return numHeads * qHeadDim; }
  int kDim() const { return numKVHeads * qHeadDim; }
  int vDim() const { return numKVHeads * vHeadDim; }
  int attDim() const { return numHeads * vHeadDim; }
};

// SwiGLU FFN, linear1 and the gate as one GEMM into rows [a | g]
struct FFN {
  float epsilon = 0.0f;
  int channels = 0;
  DeviceBuf normWeight;
  Linear linear1Gate, linear2;

  void init(const TransformerFFNDesc& desc, bool fp16) {
    testAssert(desc.useSwiGLU);
    epsilon = desc.preLN.epsilon;
    normWeight.upload(desc.preLN.name, desc.preLN.weight, false);
    testAssert(desc.linear1.outChannels == desc.linearGate.outChannels);
    channels = desc.linear1.outChannels;
    linear1Gate.initConcat({&desc.linear1, &desc.linearGate}, fp16);
    linear2.init(desc.linear2, nullptr, fp16);
    testAssert(linear2.inC == channels);
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
  DeviceBuf postScale, postBias;
  int postActivation = ACTIVATION_IDENTITY;
  Linear postConv;
};

// Offsets of the parts of a staging buffer, the same for the host and the device copies
struct Layout {
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

  int trunkChannels;
  int midChannels;
  int decoderChannels;

  Linear point;
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
  int maxQKVDim = 0, maxAttDim = 0, maxFFN = 0;
  int maxDecQDim = 0, maxDecKVDim = 0, maxDecAttDim = 0, maxDecFFN = 0;

  // Staging of the inputs and the outputs
  PinnedBuf hostIn, hostOut;
  DeviceBuf inputDev;
  // Work buffers. qkv, att and ffn are shared by the trunk and the decoder, which run one after another.
  DeviceBuf seqF, trunkH, tmp, mid, midN, qkv, att, ffn, sumSq;
  DeviceBuf pointEmb, globalProj;
  DeviceBuf pooled, poolProj, queriesF, queriesH, queriesNorm, candEmb, groundEmb, dq, policyH, policy;
  DeviceBuf outV1, pooledV, outV2, value, scoreValue, ownershipOut;

  Impl(const SparseDotsDesc& desc, bool fp16, bool mma, cudaStream_t stream_, cublasHandle_t cublas_, Logger* logger_)
    : useFP16(fp16), useMmaAttention(mma && fp16), stream(stream_), cublas(cublas_), logger(logger_) {
    trunkChannels = desc.trunkNumChannels;
    midChannels = desc.midNumChannels;
    decoderChannels = desc.decoderNumChannels;
    testAssert(desc.numPointFeatures == SparseDotsNN::NUM_POINT_FEATURES);
    testAssert(desc.numCandFeatures == SparseDotsNN::NUM_CAND_FEATURES);
    testAssert(desc.numGlobalFeatures == NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE);

    point.init(desc.pointMul, &desc.pointBias, false);
    testAssert(desc.globalToken.numChannels == trunkChannels);
    globalToken.upload(desc.globalToken.name, desc.globalToken.weights, false);
    global.init(desc.globalMul, nullptr, false);

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
          attention->init(*static_cast<const TransformerAttentionDesc*>(sub.second.get()), true, useFP16);
          maxQKVDim = std::max(maxQKVDim, attention->qkv.outC);
          maxAttDim = std::max(maxAttDim, attention->attDim());
          block->attentions.push_back(std::move(attention));
          block->isAttention.push_back(true);
        }
        else {
          testAssert(sub.first == TRANSFORMER_FFN_BLOCK_KIND);
          auto ffnLayer = std::make_unique<FFN>();
          ffnLayer->init(*static_cast<const TransformerFFNDesc*>(sub.second.get()), useFP16);
          maxFFN = std::max(maxFFN, ffnLayer->channels);
          block->ffns.push_back(std::move(ffnLayer));
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
      attention->init(desc.decoderAttention[l], false, useFP16);
      maxDecQDim = std::max(maxDecQDim, attention->qDim());
      maxDecKVDim = std::max(maxDecKVDim, attention->kv.outC);
      maxDecAttDim = std::max(maxDecAttDim, attention->attDim());
      decoderAttentions.push_back(std::move(attention));
      auto ffnLayer = std::make_unique<FFN>();
      ffnLayer->init(desc.decoderFFN[l], useFP16);
      maxDecFFN = std::max(maxDecFFN, ffnLayer->channels);
      decoderFFNs.push_back(std::move(ffnLayer));
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

  // out[rows x outC] (+)= in[rows x inC] * w, then bias and activation. inLd is the row stride of in (inC if 0).
  void gemm(const Linear& layer, const void* in, long long rows, void* out, bool accumulate = false,
            int activation = ACTIVATION_IDENTITY, int inLd = 0) {
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
      in, type, inLd > 0 ? inLd : layer.inC,
      &beta,
      out, type, layer.outC,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    if(layer.useFP16)
      sparseDotsBiasAct<half>((half*)out, layer.b.f(), rows, layer.outC, activation, stream);
    else
      sparseDotsBiasAct<float>((float*)out, layer.b.f(), rows, layer.outC, activation, stream);
  }

  // The rows of the batch (fixed for one evaluate call)
  struct Batch {
    int n = 0;          // number of positions
    int tokens = 0;     // trunk rows: 1 + points of every position
    int queries = 0;    // decoder rows: candidates + grounding of every position
    int maxTokens = 0;  // max trunk rows of a position
    int maxQueries = 0; // max decoder rows of a position
    int totalPoints = 0, totalCands = 0, totalGrounds = 0;
    vector<int> tokStartHost, qStartHost;
    // Device pointers into inputDev
    float* pointFeat; int* pointDst;
    float* global; float* boardOffset;
    int* tokStart; int* tokSeg; float* tokPos;
    float* candFeat; int* candDst;
    float* groundFeat; int* groundDst;
    int* qStart; int* qSeg; float* qPos;
  };

  void pack(const vector<const DotsSparseInput*>& inputs, const vector<int>& symmetries, Batch& b) {
    b.n = (int)inputs.size();
    b.tokStartHost.assign(b.n + 1, 0);
    b.qStartHost.assign(b.n + 1, 0);
    for(int n = 0; n < b.n; n++) {
      const DotsSparseInput* in = inputs[n];
      const int numTokens = 1 + (int)in->points.size();
      const int numQueries = (int)in->candidates.size() + (in->groundAllowed ? 1 : 0);
      b.tokStartHost[n + 1] = b.tokStartHost[n] + numTokens;
      b.qStartHost[n + 1] = b.qStartHost[n] + numQueries;
      b.maxTokens = std::max(b.maxTokens, numTokens);
      b.maxQueries = std::max(b.maxQueries, numQueries);
      b.totalPoints += (int)in->points.size();
      b.totalCands += (int)in->candidates.size();
      b.totalGrounds += in->groundAllowed ? 1 : 0;
    }
    b.tokens = b.tokStartHost[b.n];
    b.queries = b.qStartHost[b.n];

    Layout layout;
    const size_t oPointFeat = layout.reserve(sizeof(float) * b.totalPoints * SparseDotsNN::NUM_POINT_FEATURES);
    const size_t oPointDst = layout.reserve(sizeof(int) * b.totalPoints);
    const size_t oGlobal = layout.reserve(sizeof(float) * b.n * NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE);
    const size_t oBoardOffset = layout.reserve(sizeof(float) * b.n);
    const size_t oTokStart = layout.reserve(sizeof(int) * (b.n + 1));
    const size_t oTokSeg = layout.reserve(sizeof(int) * b.tokens);
    const size_t oTokPos = layout.reserve(sizeof(float) * 3 * b.tokens);
    const size_t oCandFeat = layout.reserve(sizeof(float) * b.totalCands * SparseDotsNN::NUM_CAND_FEATURES);
    const size_t oCandDst = layout.reserve(sizeof(int) * b.totalCands);
    const size_t oGroundFeat = layout.reserve(sizeof(float) * b.totalGrounds);
    const size_t oGroundDst = layout.reserve(sizeof(int) * b.totalGrounds);
    const size_t oQStart = layout.reserve(sizeof(int) * (b.n + 1));
    const size_t oQSeg = layout.reserve(sizeof(int) * b.queries);
    const size_t oQPos = layout.reserve(sizeof(float) * 3 * b.queries);
    hostIn.ensure(layout.size);

    char* h = hostIn.ptr;
    float* pointFeat = (float*)(h + oPointFeat);
    int* pointDst = (int*)(h + oPointDst);
    float* global = (float*)(h + oGlobal);
    float* boardOffset = (float*)(h + oBoardOffset);
    int* tokSeg = (int*)(h + oTokSeg);
    float* tokPos = (float*)(h + oTokPos);
    float* candFeat = (float*)(h + oCandFeat);
    int* candDst = (int*)(h + oCandDst);
    float* groundFeat = (float*)(h + oGroundFeat);
    int* groundDst = (int*)(h + oGroundDst);
    int* qSeg = (int*)(h + oQSeg);
    float* qPos = (float*)(h + oQPos);
    std::memcpy(h + oTokStart, b.tokStartHost.data(), sizeof(int) * (b.n + 1));
    std::memcpy(h + oQStart, b.qStartHost.data(), sizeof(int) * (b.n + 1));

    int pointRow = 0, candRow = 0, groundRow = 0;
    for(int n = 0; n < b.n; n++) {
      const DotsSparseInput& in = *inputs[n];
      const int symmetry = symmetries[n];
      const bool transpose = SymmetryHelpers::isTranspose(symmetry);
      const int xSize = transpose ? in.ySize : in.xSize;
      const int ySize = transpose ? in.xSize : in.ySize;
      const int numPoints = (int)in.points.size();
      const int numCands = (int)in.candidates.size();
      const int tokenBase = b.tokStartHost[n];
      const int queryBase = b.qStartHost[n];

      float* g = global + (size_t)n * NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE;
      std::copy(in.global, in.global + NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE, g);
      if(transpose) {
        std::swap(
          g[static_cast<int>(NNInputs::DotsSparseGlobalFeature::XSize_6)],
          g[static_cast<int>(NNInputs::DotsSparseGlobalFeature::YSize_7)]);
      }
      boardOffset[n] = sqrtf((float)in.xSize * (float)in.ySize) - 14.0f;

      // Trunk rows: GLOBAL (no position) then the points
      tokSeg[tokenBase] = n;
      tokPos[3 * tokenBase] = 0.0f;
      tokPos[3 * tokenBase + 1] = 0.0f;
      tokPos[3 * tokenBase + 2] = 0.0f;
      for(int i = 0; i < numPoints; i++) {
        const auto& p = in.points[i];
        int x, y;
        DotsSparseInput::applySymmetry(p.x, p.y, in.xSize, in.ySize, symmetry, x, y);
        SparseDotsNN::pointFeaturesAt(p, x, y, xSize, ySize, pointFeat + (size_t)(pointRow + i) * SparseDotsNN::NUM_POINT_FEATURES);
        const int row = tokenBase + 1 + i;
        pointDst[pointRow + i] = row;
        tokSeg[row] = n;
        tokPos[3 * row] = (float)x;
        tokPos[3 * row + 1] = (float)y;
        tokPos[3 * row + 2] = 1.0f;
      }

      // Decoder rows: the point candidates then the grounding (no position)
      for(int i = 0; i < numCands; i++) {
        const auto& cnd = in.candidates[i];
        int x, y;
        DotsSparseInput::applySymmetry(cnd.x, cnd.y, in.xSize, in.ySize, symmetry, x, y);
        SparseDotsNN::candidateFeaturesAt(cnd, x, y, xSize, ySize, candFeat + (size_t)(candRow + i) * SparseDotsNN::NUM_CAND_FEATURES);
        const int row = queryBase + i;
        candDst[candRow + i] = row;
        qSeg[row] = n;
        qPos[3 * row] = (float)x;
        qPos[3 * row + 1] = (float)y;
        qPos[3 * row + 2] = 1.0f;
      }
      if(in.groundAllowed) {
        const int row = queryBase + numCands;
        groundFeat[groundRow] = SparseDotsNN::groundFeature(in);
        groundDst[groundRow] = row;
        groundRow++;
        qSeg[row] = n;
        qPos[3 * row] = 0.0f;
        qPos[3 * row + 1] = 0.0f;
        qPos[3 * row + 2] = 0.0f;
      }

      pointRow += numPoints;
      candRow += numCands;
    }
    testAssert(pointRow == b.totalPoints && candRow == b.totalCands && groundRow == b.totalGrounds);

    inputDev.ensure(layout.size);
    char* d = (char*)inputDev.ptr;
    CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(d, h, layout.size, cudaMemcpyHostToDevice, stream));
    b.pointFeat = (float*)(d + oPointFeat);
    b.pointDst = (int*)(d + oPointDst);
    b.global = (float*)(d + oGlobal);
    b.boardOffset = (float*)(d + oBoardOffset);
    b.tokStart = (int*)(d + oTokStart);
    b.tokSeg = (int*)(d + oTokSeg);
    b.tokPos = (float*)(d + oTokPos);
    b.candFeat = (float*)(d + oCandFeat);
    b.candDst = (int*)(d + oCandDst);
    b.groundFeat = (float*)(d + oGroundFeat);
    b.groundDst = (int*)(d + oGroundDst);
    b.qStart = (int*)(d + oQStart);
    b.qSeg = (int*)(d + oQSeg);
    b.qPos = (float*)(d + oQPos);
  }

  void ensureBuffers(const Batch& b) {
    const size_t tokens = (size_t)b.tokens;
    const size_t queries = (size_t)b.queries;
    const size_t actBytes = useFP16 ? sizeof(half) : sizeof(float);
    const size_t fb = sizeof(float);
    const int c = trunkChannels;
    const int m = midChannels;
    const int cd = decoderChannels;

    seqF.ensure(tokens * c * fb);
    if(useFP16)
      trunkH.ensure(tokens * c * actBytes);
    tmp.ensure(tokens * c * actBytes);
    mid.ensure(tokens * m * actBytes);
    midN.ensure(tokens * m * actBytes);
    qkv.ensure(std::max(tokens * maxQKVDim, tokens * maxDecKVDim) * actBytes);
    att.ensure(std::max(tokens * maxAttDim, queries * maxDecAttDim) * actBytes);
    ffn.ensure(std::max(tokens * 2 * maxFFN, queries * 2 * maxDecFFN) * actBytes);
    sumSq.ensure((size_t)b.n * fb);

    pointEmb.ensure((size_t)b.totalPoints * c * fb);
    globalProj.ensure((size_t)b.n * c * fb);

    pooled.ensure((size_t)b.n * 3 * c * fb);
    poolProj.ensure((size_t)b.n * cd * fb);
    queriesF.ensure(queries * cd * fb);
    if(useFP16) {
      queriesH.ensure(queries * cd * actBytes);
      policyH.ensure(queries * decoderOut.outC * actBytes);
    }
    queriesNorm.ensure(queries * cd * actBytes);
    candEmb.ensure((size_t)b.totalCands * cd * fb);
    groundEmb.ensure((size_t)b.totalGrounds * cd * fb);
    dq.ensure(queries * maxDecQDim * actBytes);
    policy.ensure(queries * decoderOut.outC * fb);

    outV1.ensure(tokens * v1.outC * fb);
    pooledV.ensure((size_t)b.n * 3 * v1.outC * fb);
    outV2.ensure((size_t)b.n * v2.outC * fb);
    value.ensure((size_t)b.n * v3.outC * fb);
    scoreValue.ensure((size_t)b.n * sv3.outC * fb);
    ownershipOut.ensure(tokens * fb);
  }

  // Attention of the rows of every position (Q with qStride) to the keys of the same position (K and V with kvStride)
  template <typename T>
  void attention(const Attention& layer, const T* Q, const T* K, const T* V, T* out, const int* qStart, const int* kvStart,
                 int numSegs, int maxQLen, int qStride, int kvStride) {
    bool done = false;
    if constexpr(std::is_same<T, half>::value) {
      if(useMmaAttention && layer.qHeadDim == layer.vHeadDim) {
        done = customCudaFlashAttentionMmaVarlen(
          Q, K, V, out, qStart, kvStart, numSegs, maxQLen, layer.numHeads, layer.numKVHeads, layer.qHeadDim,
          layer.vHeadDim, qStride, kvStride, stream);
        if(done && !loggedMma) {
          loggedMma = true;
          if(logger != nullptr)
            logger->write("Sparse Dots CUDA: using tensor-core mma flash attention kernel");
        }
      }
    }
    if(!done)
      sparseDotsAttention<T>(Q, K, V, out, qStart, kvStart, numSegs, maxQLen, layer.numHeads, layer.numKVHeads,
                             layer.qHeadDim, layer.vHeadDim, qStride, kvStride, stream);
  }

  template <typename T>
  void trunkAttention(const Attention& layer, const Batch& b) {
    const long long rows = b.tokens;
    T* midT = (T*)mid.ptr;
    T* midNT = (T*)midN.ptr;
    T* qkvT = (T*)qkv.ptr;
    T* attT = (T*)att.ptr;
    const int stride = layer.qkv.outC;
    sparseDotsRMSNorm<T>(midT, midNT, layer.normWeight.f(), rows, midChannels, layer.epsilon, stream);
    gemm(layer.qkv, midNT, rows, qkvT);
    T* qT = qkvT;
    T* kT = qkvT + layer.qDim();
    T* vT = kT + layer.kDim();
    sparseDotsApplyRopeQK<T>(qT, kT, layer.ropeFreqs.f(), b.tokPos, (int)rows, layer.numHeads, layer.numKVHeads, layer.qHeadDim,
                             stride, stride, stream);
    attention<T>(layer, qT, kT, vT, attT, b.tokStart, b.tokStart, b.n, b.maxTokens, stride, stride);
    gemm(layer.out, attT, rows, midT, true);
  }

  // x += FFN(x)
  template <typename T>
  void ffnResidual(const FFN& layer, T* x, T* xNorm, long long rows) {
    T* ab = (T*)ffn.ptr;
    sparseDotsRMSNorm<T>(x, xNorm, layer.normWeight.f(), rows, layer.linear1Gate.inC, layer.epsilon, stream);
    gemm(layer.linear1Gate, xNorm, rows, ab);
    sparseDotsSwiGLU<T>(ab, rows, layer.channels, stream);
    gemm(layer.linear2, ab, rows, x, true, ACTIVATION_IDENTITY, 2 * layer.channels);
  }

  // Runs the trunk in place, ends with the tip norm
  template <typename T>
  void runTrunk(T* trunk, const Batch& b) {
    const long long rows = b.tokens;
    T* tmpT = (T*)tmp.ptr;
    T* midT = (T*)mid.ptr;
    T* midNT = (T*)midN.ptr;
    for(const auto& block : blocks) {
      sparseDotsScaleBiasAct<T>(trunk, tmpT, block->preScale.f(), block->preBias.f(), rows, trunkChannels,
                                block->preActivation, stream);
      gemm(block->preConv, tmpT, rows, midT);
      size_t attentionIdx = 0;
      size_t ffnIdx = 0;
      for(bool isAttention : block->isAttention) {
        if(isAttention)
          trunkAttention<T>(*block->attentions[attentionIdx++], b);
        else
          ffnResidual<T>(*block->ffns[ffnIdx++], midT, midNT, rows);
      }
      sparseDotsScaleBiasAct<T>(midT, midNT, block->postScale.f(), block->postBias.f(), rows, midChannels,
                                block->postActivation, stream);
      gemm(block->postConv, midNT, rows, trunk, true);
    }
    sparseDotsSpatialRMSNorm<T>(trunk, trunk, tipGamma.f(), tipBeta.f(), b.tokSeg, b.tokStart, b.n, b.tokens, trunkChannels,
                                tipEpsilon, tipActivation, sumSq.f(), stream);
  }

  // Runs the decoder layers on the embedded queries, writes the logits to policy (FP32) or policyH (FP16)
  template <typename T>
  void runDecoder(const T* trunk, T* query, const Batch& b) {
    const long long tokens = b.tokens;
    const long long numQueries = b.queries;
    const int cd = decoderChannels;
    T* queryNorm = (T*)queriesNorm.ptr;
    T* dqT = (T*)dq.ptr;
    T* kvT = (T*)qkv.ptr;
    T* attT = (T*)att.ptr;
    for(size_t l = 0; l < decoderAttentions.size(); l++) {
      const Attention& layer = *decoderAttentions[l];
      const int kvStride = layer.kv.outC;
      sparseDotsRMSNorm<T>(query, queryNorm, layer.normWeight.f(), numQueries, cd, layer.epsilon, stream);
      gemm(layer.q, queryNorm, numQueries, dqT);
      gemm(layer.kv, trunk, tokens, kvT);
      sparseDotsApplyRope<T>(dqT, layer.ropeFreqs.f(), b.qPos, (int)numQueries, layer.numHeads, layer.numKVHeads,
                             layer.qHeadDim, layer.qDim(), stream);
      sparseDotsApplyRope<T>(kvT, layer.ropeFreqs.f(), b.tokPos, (int)tokens, layer.numKVHeads, layer.numKVHeads,
                             layer.qHeadDim, kvStride, stream);
      attention<T>(layer, dqT, kvT, kvT + layer.kDim(), attT, b.qStart, b.tokStart, b.n, b.maxQueries, layer.qDim(), kvStride);
      gemm(layer.out, attT, numQueries, query, true);
      ffnResidual<T>(*decoderFFNs[l], query, queryNorm, numQueries);
    }
    sparseDotsRMSNorm<T>(query, queryNorm, decoderNormWeight.f(), numQueries, cd, decoderNormEpsilon, stream);
    gemm(decoderOut, queryNorm, numQueries, std::is_same<T, half>::value ? policyH.ptr : policy.ptr);
  }

  void evaluate(
    const vector<const DotsSparseInput*>& inputs, const vector<int>& symmetries, vector<SparseDotsNN::RawOutput>& outputs
  ) {
    testAssert(!inputs.empty());
    testAssert(inputs.size() == symmetries.size());
    Batch b;
    const auto hostStart = std::chrono::steady_clock::now();
    pack(inputs, symmetries, b);
    ensureBuffers(b);
    const double hostMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hostStart).count();
    mark("start");
    const long long tokens = b.tokens;
    const long long queries = b.queries;
    const int c = trunkChannels;
    const int cd = decoderChannels;

    // Embeddings in FP32 --------------------------------------------------------------------------------------------
    float* seq = seqF.f();
    gemm(point, b.pointFeat, b.totalPoints, pointEmb.ptr);
    sparseDotsScatterRows(pointEmb.f(), seq, b.pointDst, b.totalPoints, c, stream);
    gemm(global, b.global, b.n, globalProj.ptr);
    sparseDotsAddGlobal(seq, globalToken.f(), globalProj.f(), b.tokSeg, b.tokStart, b.tokens, c, stream);
    mark("embeddings");

    // Trunk -----------------------------------------------------------------------------------------------------------
    if(useFP16) {
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
    if(queries > 0) {
      sparseDotsPool(trunk, b.tokStart, b.boardOffset, pooled.f(), b.n, c, false, stream);
      gemm(pool, pooled.ptr, b.n, poolProj.ptr);
      // Every query row is a candidate or a grounding row, so the scatters fill all of them
      float* queryF = queriesF.f();
      gemm(cand, b.candFeat, b.totalCands, candEmb.ptr);
      sparseDotsScatterRows(candEmb.f(), queryF, b.candDst, b.totalCands, cd, stream);
      gemm(ground, b.groundFeat, b.totalGrounds, groundEmb.ptr);
      sparseDotsScatterRows(groundEmb.f(), queryF, b.groundDst, b.totalGrounds, cd, stream);
      sparseDotsAddRowVector(queryF, poolProj.f(), b.qSeg, b.queries, cd, stream);
      if(useFP16) {
        sparseDotsCopyToHalf(queryF, (half*)queriesH.ptr, queries * cd, stream);
        runDecoder<half>((const half*)trunkH.ptr, (half*)queriesH.ptr, b);
        sparseDotsCopyFromHalf((const half*)policyH.ptr, policy.f(), queries * decoderOut.outC, stream);
      }
      else {
        runDecoder<float>(trunk, queryF, b);
      }
    }
    mark("decoder");

    // Value head in FP32 ----------------------------------------------------------------------------------------------
    gemm(v1, trunk, tokens, outV1.ptr, false, v1Activation);
    sparseDotsPool(outV1.f(), b.tokStart, b.boardOffset, pooledV.f(), b.n, v1.outC, true, stream);
    gemm(v2, pooledV.ptr, b.n, outV2.ptr, false, v2Activation);
    gemm(v3, outV2.ptr, b.n, value.ptr);
    gemm(sv3, outV2.ptr, b.n, scoreValue.ptr);
    gemm(ownership, outV1.ptr, tokens, ownershipOut.ptr);
    CUDA_ERR("SparseDotsCudaModel", cudaPeekAtLastError());
    mark("value head");

    // Outputs ---------------------------------------------------------------------------------------------------------
    const int policyC = decoderOut.outC;
    Layout layout;
    const size_t oPolicy = layout.reserve(sizeof(float) * queries * policyC);
    const size_t oValue = layout.reserve(sizeof(float) * b.n * 3);
    const size_t oScore = layout.reserve(sizeof(float) * b.n * 6);
    const size_t oOwnership = layout.reserve(sizeof(float) * tokens);
    hostOut.ensure(layout.size);
    const float* policyHost = (const float*)(hostOut.ptr + oPolicy);
    const float* valueHost = (const float*)(hostOut.ptr + oValue);
    const float* scoreHost = (const float*)(hostOut.ptr + oScore);
    const float* ownershipHost = (const float*)(hostOut.ptr + oOwnership);
    if(queries > 0)
      CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(hostOut.ptr + oPolicy, policy.ptr, sizeof(float) * queries * policyC, cudaMemcpyDeviceToHost, stream));
    CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(hostOut.ptr + oValue, value.ptr, sizeof(float) * b.n * 3, cudaMemcpyDeviceToHost, stream));
    CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(hostOut.ptr + oScore, scoreValue.ptr, sizeof(float) * b.n * 6, cudaMemcpyDeviceToHost, stream));
    CUDA_ERR("SparseDotsCudaModel", cudaMemcpyAsync(hostOut.ptr + oOwnership, ownershipOut.ptr, sizeof(float) * tokens, cudaMemcpyDeviceToHost, stream));
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
      const size_t queryBase = (size_t)b.qStartHost[n];
      out.policy.resize((size_t)numQueries * 2);
      for(int i = 0; i < numQueries; i++) {
        out.policy[(size_t)i * 2] = policyHost[(queryBase + i) * policyC];
        out.policy[(size_t)i * 2 + 1] = policyHost[(queryBase + i) * policyC + 1];
      }
      for(int i = 0; i < 3; i++)
        out.value[i] = valueHost[(size_t)n * 3 + i];
      for(int i = 0; i < 6; i++)
        out.scoreValue[i] = scoreHost[(size_t)n * 6 + i];
      const size_t pointBase = (size_t)b.tokStartHost[n] + 1;
      out.ownership.assign(ownershipHost + pointBase, ownershipHost + pointBase + in.points.size());
    }
  }
};

SparseDotsCudaModel::SparseDotsCudaModel(
  const SparseDotsDesc& desc, bool useFP16, bool useMmaAttention, cudaStream_t stream, cublasHandle_t cublas, Logger* logger)
  : impl(std::make_unique<Impl>(desc, useFP16, useMmaAttention, stream, cublas, logger)) {}

SparseDotsCudaModel::~SparseDotsCudaModel() = default;

void SparseDotsCudaModel::evaluate(
  const vector<const DotsSparseInput*>& inputs, const vector<int>& symmetries, vector<SparseDotsNN::RawOutput>& outputs
) {
  impl->evaluate(inputs, symmetries, outputs);
}
