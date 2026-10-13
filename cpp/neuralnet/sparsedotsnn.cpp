#include "../neuralnet/sparsedotsnn.h"

#include <cmath>

#if defined(USE_EIGEN_BACKEND) || defined(KATAGO_HAS_EIGEN)
#define SPARSE_DOTS_USE_EIGEN
#include <Eigen/Dense>
#endif

#include "../core/test.h"

using namespace std;

namespace {

constexpr float COORD_SCALE = 1.0f / 40.0f;
constexpr float COUNT_SCALE = 1.0f / 20.0f;
constexpr int NUM_MOVES_AGO = 5;

// out[rows x outC] = in[rows x inC] * w[inC x outC], all row-major
void matmul(const float* in, int rows, int inC, const float* w, int outC, float* out) {
#ifdef SPARSE_DOTS_USE_EIGEN
  using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const RowMatrix> a(in, rows, inC);
  Eigen::Map<const RowMatrix> b(w, inC, outC);
  Eigen::Map<RowMatrix> c(out, rows, outC);
  c.noalias() = a * b;
#else
  for(int r = 0; r < rows; r++) {
    float* o = out + (size_t)r * outC;
    std::fill(o, o + outC, 0.0f);
    const float* x = in + (size_t)r * inC;
    for(int i = 0; i < inC; i++) {
      const float a = x[i];
      if(a == 0.0f)
        continue;
      const float* wi = w + (size_t)i * outC;
      for(int c = 0; c < outC; c++)
        o[c] += a * wi[c];
    }
  }
#endif
}

float activate(int activation, float x) {
  switch(activation) {
    case ACTIVATION_IDENTITY: return x;
    case ACTIVATION_RELU: return x > 0.0f ? x : 0.0f;
    case ACTIVATION_SILU: return x / (1.0f + expf(-x));
    case ACTIVATION_MISH: return x * tanhf(x < 20.0f ? log1pf(expf(x)) : x);
    default: throw StringError("Sparse Dots CPU: unsupported activation " + Global::intToString(activation));
  }
}

void activateAll(int activation, float* x, size_t n) {
  if(activation == ACTIVATION_IDENTITY)
    return;
  for(size_t i = 0; i < n; i++)
    x[i] = activate(activation, x[i]);
}

// A linear layer with weights [in x out] row-major and an optional bias
struct Linear {
  int inC = 0;
  int outC = 0;
  vector<float> w;
  vector<float> b;

  void init(const MatMulLayerDesc& desc) {
    inC = desc.inChannels;
    outC = desc.outChannels;
    w = desc.weights;
  }
  void init(const MatMulLayerDesc& desc, const MatBiasLayerDesc& bias) {
    init(desc);
    testAssert(bias.numChannels == outC);
    b = bias.weights;
  }
  void initConv1x1(const ConvLayerDesc& desc) {
    testAssert(desc.convXSize == 1 && desc.convYSize == 1);
    inC = desc.inChannels;
    outC = desc.outChannels;
    // ConvLayerDesc stores [oc][ic]
    w.resize((size_t)inC * outC);
    for(int ic = 0; ic < inC; ic++)
      for(int oc = 0; oc < outC; oc++)
        w[(size_t)ic * outC + oc] = desc.weights[(size_t)oc * inC + ic];
  }
  void apply(const float* in, int rows, float* out) const {
    matmul(in, rows, inC, w.data(), outC, out);
    if(!b.empty()) {
      for(int r = 0; r < rows; r++)
        for(int c = 0; c < outC; c++)
          out[(size_t)r * outC + c] += b[c];
    }
  }
};

// Per-channel scale and bias, then the activation (the norms of the fixscale trunk are not batch norms)
struct ScaleBiasAct {
  vector<float> scale;
  vector<float> bias;
  int activation = ACTIVATION_IDENTITY;

  void init(const BatchNormLayerDesc& bn, const ActivationLayerDesc& act) {
    scale = bn.mergedScale;
    bias = bn.mergedBias;
    activation = act.activation;
  }
  void apply(const float* in, int rows, float* out) const {
    const int c = (int)scale.size();
    for(int r = 0; r < rows; r++)
      for(int i = 0; i < c; i++)
        out[(size_t)r * c + i] = activate(activation, in[(size_t)r * c + i] * scale[i] + bias[i]);
  }
};

struct RowRMSNorm {
  vector<float> weight;
  float epsilon = 1e-6f;

  void init(const TransformerRMSNormDesc& desc) {
    weight = desc.weight;
    epsilon = desc.epsilon;
  }
  void apply(const float* in, int rows, float* out) const {
    const int c = (int)weight.size();
    for(int r = 0; r < rows; r++) {
      const float* x = in + (size_t)r * c;
      float sumSq = 0.0f;
      for(int i = 0; i < c; i++)
        sumSq += x[i] * x[i];
      const float inv = 1.0f / sqrtf(sumSq / (float)c + epsilon);
      for(int i = 0; i < c; i++)
        out[(size_t)r * c + i] = x[i] * inv * weight[i];
    }
  }
};

struct Attention {
  int numHeads = 0;
  int numKVHeads = 0;
  int qHeadDim = 0;
  int vHeadDim = 0;
  RowRMSNorm norm;
  Linear q, k, v, out;
  vector<float> ropeFreqs; // [kvHeads][pairs][2]

  void init(const TransformerAttentionDesc& desc) {
    testAssert(desc.useRope && desc.learnableRope);
    numHeads = desc.numHeads;
    numKVHeads = desc.numKVHeads;
    qHeadDim = desc.qHeadDim;
    vHeadDim = desc.vHeadDim;
    norm.init(desc.preLN);
    q.init(desc.qProj);
    k.init(desc.kProj);
    v.init(desc.vProj);
    out.init(desc.outProj);
    ropeFreqs = desc.ropeFreqs;
  }
  int numPairs() const { return qHeadDim / 2; }
};

struct FFN {
  RowRMSNorm norm;
  Linear linear1, gate, linear2;

  void init(const TransformerFFNDesc& desc) {
    testAssert(desc.useSwiGLU);
    norm.init(desc.preLN);
    linear1.init(desc.linear1);
    gate.init(desc.linearGate);
    linear2.init(desc.linear2);
  }
  // x += FFN(x)
  void applyResidual(float* x, int rows) const {
    const int c = linear1.inC;
    const int f = linear1.outC;
    vector<float> xn((size_t)rows * c);
    vector<float> a((size_t)rows * f);
    vector<float> g((size_t)rows * f);
    vector<float> o((size_t)rows * c);
    norm.apply(x, rows, xn.data());
    linear1.apply(xn.data(), rows, a.data());
    gate.apply(xn.data(), rows, g.data());
    for(size_t i = 0; i < a.size(); i++)
      a[i] = a[i] / (1.0f + expf(-a[i])) * g[i];
    linear2.apply(a.data(), rows, o.data());
    for(size_t i = 0; i < o.size(); i++)
      x[i] += o[i];
  }
};

struct TrunkBlock {
  ScaleBiasAct pre;
  Linear preConv;
  // Alternating attention and FFN sub-blocks
  vector<Attention> attentions;
  vector<FFN> ffns;
  vector<bool> isAttention;
  ScaleBiasAct post;
  Linear postConv;
};

// cos/sin tables [rows][kvHeads][pairs] for weighted position sets
struct RopeTables {
  vector<float> cos;
  vector<float> sin;
};

}  // namespace

static void edgeFeatures(const DotsSparseInput& input, float x, float y, float* out) {
  out[0] = x * COORD_SCALE;
  out[1] = y * COORD_SCALE;
  out[2] = ((float)input.xSize - 1.0f - x) * COORD_SCALE;
  out[3] = ((float)input.ySize - 1.0f - y) * COORD_SCALE;
}

static void movesAgoOneHot(int movesAgo, float* out) {
  for(int i = 0; i < NUM_MOVES_AGO; i++)
    out[i] = movesAgo == i + 1 ? 1.0f : 0.0f;
}

void SparseDotsNN::pointFeatures(const DotsSparseInput& input, int i, float* out) {
  const auto& p = input.points[i];
  out[0] = p.owner;
  edgeFeatures(input, p.x, p.y, out + 1);
  out[5] = p.grounded;
  movesAgoOneHot(p.movesAgo, out + 6);
  // Facts of the areas whose boundary has the dot
  out[11] = p.areaCount;
  out[12] = p.insideMy * COUNT_SCALE;
  out[13] = p.insideOpp * COUNT_SCALE;
  movesAgoOneHot(p.areaMovesAgo, out + 14);
}

void SparseDotsNN::candidateFeatures(const DotsSparseInput& input, int i, float* out) {
  const auto& cnd = input.candidates[i];
  edgeFeatures(input, cnd.x, cnd.y, out);
  out[4] = cnd.gainMy * COUNT_SCALE;
  out[5] = cnd.gainOpp * COUNT_SCALE;
  out[6] = cnd.inMyHouse;
  out[7] = cnd.inOppHouse;
}

float SparseDotsNN::groundFeature(const DotsSparseInput& input) {
  return input.groundResult * COUNT_SCALE;
}

struct SparseDotsNN::Model::Impl {
  int trunkChannels;
  int midChannels;
  int decoderChannels;

  Linear point;
  vector<float> globalToken;
  Linear global;

  vector<TrunkBlock> blocks;
  vector<float> tipGamma;
  vector<float> tipBeta;
  float tipEpsilon;
  int tipActivation;

  Linear cand;
  Linear ground;
  Linear pool;
  vector<Attention> decoderAttentions;
  vector<FFN> decoderFFNs;
  RowRMSNorm decoderNorm;
  Linear decoderOut;

  Linear v1;
  int v1Activation;
  Linear v2;
  int v2Activation;
  Linear v3;
  Linear sv3;
  Linear ownership;

  explicit Impl(const SparseDotsDesc& desc) {
    trunkChannels = desc.trunkNumChannels;
    midChannels = desc.midNumChannels;
    decoderChannels = desc.decoderNumChannels;

    point.init(desc.pointMul, desc.pointBias);
    globalToken = desc.globalToken.weights;
    global.init(desc.globalMul);

    for(const auto& blockPair : desc.blocks) {
      const auto& nested = *static_cast<const NestedBottleneckResidualBlockDesc*>(blockPair.second.get());
      TrunkBlock block;
      block.pre.init(nested.preBN, nested.preActivation);
      block.preConv.initConv1x1(nested.preConv);
      for(const auto& sub : nested.blocks) {
        if(sub.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
          block.attentions.emplace_back();
          block.attentions.back().init(*static_cast<const TransformerAttentionDesc*>(sub.second.get()));
          block.isAttention.push_back(true);
        }
        else {
          testAssert(sub.first == TRANSFORMER_FFN_BLOCK_KIND);
          block.ffns.emplace_back();
          block.ffns.back().init(*static_cast<const TransformerFFNDesc*>(sub.second.get()));
          block.isAttention.push_back(false);
        }
      }
      block.post.init(nested.postBN, nested.postActivation);
      block.postConv.initConv1x1(nested.postConv);
      blocks.push_back(std::move(block));
    }
    tipGamma = desc.trunkTipRMSNorm.gamma;
    tipBeta = desc.trunkTipRMSNorm.beta;
    tipEpsilon = desc.trunkTipRMSNorm.epsilon;
    tipActivation = desc.trunkTipActivation.activation;

    cand.init(desc.candMul, desc.candBias);
    ground.init(desc.groundMul, desc.groundBias);
    pool.init(desc.poolMul);
    for(const auto& attention : desc.decoderAttention) {
      decoderAttentions.emplace_back();
      decoderAttentions.back().init(attention);
    }
    for(const auto& ffn : desc.decoderFFN) {
      decoderFFNs.emplace_back();
      decoderFFNs.back().init(ffn);
    }
    decoderNorm.init(desc.decoderFinalNorm);
    decoderOut.init(desc.decoderOutMul);

    v1.init(desc.v1Mul, desc.v1Bias);
    v1Activation = desc.v1Activation.activation;
    v2.init(desc.v2Mul, desc.v2Bias);
    v2Activation = desc.v2Activation.activation;
    v3.init(desc.v3Mul, desc.v3Bias);
    sv3.init(desc.sv3Mul, desc.sv3Bias);
    ownership.init(desc.ownershipMul);
  }

  // Tables of the trunk tokens [GLOBAL][POINT x P] for the frequencies of a layer, GLOBAL has no position
  void tokenTables(const Attention& layer, const DotsSparseInput& input, RopeTables& tables) const {
    const int kvHeads = layer.numKVHeads;
    const int pairs = layer.numPairs();
    const int hp = kvHeads * pairs;
    const int numPoints = (int)input.points.size();
    const int seqLen = 1 + numPoints;
    tables.cos.assign((size_t)seqLen * hp, 0.0f);
    tables.sin.assign((size_t)seqLen * hp, 0.0f);
    const float* freqs = layer.ropeFreqs.data();
    for(int t = 0; t < numPoints; t++) {
      const float x = input.points[t].x;
      const float y = input.points[t].y;
      for(int i = 0; i < hp; i++) {
        const float angle = x * freqs[2 * i] + y * freqs[2 * i + 1];
        tables.cos[(size_t)(1 + t) * hp + i] = cosf(angle);
        tables.sin[(size_t)(1 + t) * hp + i] = sinf(angle);
      }
    }
  }

  // Tables of the candidates [point candidates][grounding if allowed], grounding has no position
  static void candidateTables(const Attention& layer, const DotsSparseInput& input, int numQueries, RopeTables& tables) {
    const int hp = layer.numKVHeads * layer.numPairs();
    tables.cos.assign((size_t)numQueries * hp, 0.0f);
    tables.sin.assign((size_t)numQueries * hp, 0.0f);
    const float* freqs = layer.ropeFreqs.data();
    for(int t = 0; t < (int)input.candidates.size(); t++) {
      const float x = input.candidates[t].x;
      const float y = input.candidates[t].y;
      for(int i = 0; i < hp; i++) {
        const float angle = x * freqs[2 * i] + y * freqs[2 * i + 1];
        tables.cos[(size_t)t * hp + i] = cosf(angle);
        tables.sin[(size_t)t * hp + i] = sinf(angle);
      }
    }
  }

  // Rotates the pairs (2p, 2p+1) of every head of the rows; heads map to kv heads h * kvHeads / heads
  static void rotate(float* data, int rows, int heads, int headDim, int kvHeads, const RopeTables& tables) {
    const int pairs = headDim / 2;
    const int hp = kvHeads * pairs;
    for(int r = 0; r < rows; r++) {
      for(int h = 0; h < heads; h++) {
        const int kvh = h * kvHeads / heads;
        float* x = data + (size_t)r * heads * headDim + (size_t)h * headDim;
        const float* c = tables.cos.data() + (size_t)r * hp + (size_t)kvh * pairs;
        const float* s = tables.sin.data() + (size_t)r * hp + (size_t)kvh * pairs;
        for(int p = 0; p < pairs; p++) {
          const float x0 = x[2 * p];
          const float x1 = x[2 * p + 1];
          x[2 * p] = x0 * c[p] - x1 * s[p];
          x[2 * p + 1] = x0 * s[p] + x1 * c[p];
        }
      }
    }
  }

  // Attention of the normalized queries [numQueries x Cq] to the keys/values source [seqLen x Ckv].
  // Returns the output projection [numQueries x Cq] in result.
  static void attend(
    const Attention& layer,
    const float* queriesNorm, int numQueries, const RopeTables& queryTables,
    const float* source, int seqLen, const RopeTables& keyTables,
    vector<float>& result
  ) {
    const int heads = layer.numHeads;
    const int kvHeads = layer.numKVHeads;
    const int dq = layer.qHeadDim;
    const int dv = layer.vHeadDim;
    vector<float> q((size_t)numQueries * heads * dq);
    vector<float> k((size_t)seqLen * kvHeads * dq);
    vector<float> v((size_t)seqLen * kvHeads * dv);
    layer.q.apply(queriesNorm, numQueries, q.data());
    layer.k.apply(source, seqLen, k.data());
    layer.v.apply(source, seqLen, v.data());
    rotate(q.data(), numQueries, heads, dq, kvHeads, queryTables);
    rotate(k.data(), seqLen, kvHeads, dq, kvHeads, keyTables);

    const float scale = 1.0f / sqrtf((float)dq);
    vector<float> attnOut((size_t)numQueries * heads * dv, 0.0f);
    vector<float> scores(seqLen);
    for(int i = 0; i < numQueries; i++) {
      for(int h = 0; h < heads; h++) {
        const int kvh = h * kvHeads / heads;
        const float* qi = q.data() + (size_t)i * heads * dq + (size_t)h * dq;
        float maxScore = -1e30f;
        for(int j = 0; j < seqLen; j++) {
          const float* kj = k.data() + (size_t)j * kvHeads * dq + (size_t)kvh * dq;
          float dot = 0.0f;
          for(int d = 0; d < dq; d++)
            dot += qi[d] * kj[d];
          scores[j] = dot * scale;
          maxScore = std::max(maxScore, scores[j]);
        }
        float sum = 0.0f;
        for(int j = 0; j < seqLen; j++) {
          scores[j] = expf(scores[j] - maxScore);
          sum += scores[j];
        }
        float* o = attnOut.data() + (size_t)i * heads * dv + (size_t)h * dv;
        for(int j = 0; j < seqLen; j++) {
          const float weight = scores[j] / sum;
          const float* vj = v.data() + (size_t)j * kvHeads * dv + (size_t)kvh * dv;
          for(int d = 0; d < dv; d++)
            o[d] += weight * vj[d];
        }
      }
    }
    result.resize((size_t)numQueries * layer.out.outC);
    layer.out.apply(attnOut.data(), numQueries, result.data());
  }

  void evaluate(const DotsSparseInput& input, RawOutput& result) const {
    const int numPoints = (int)input.points.size();
    const int numCands = (int)input.candidates.size();
    const int seqLen = 1 + numPoints;
    const int c = trunkChannels;
    const float xSize = (float)input.xSize;
    const float ySize = (float)input.ySize;

    // Embeddings ------------------------------------------------------------------------------------------------
    vector<float> trunk((size_t)seqLen * c);
    std::copy(globalToken.begin(), globalToken.end(), trunk.begin());
    if(numPoints > 0) {
      vector<float> features((size_t)numPoints * NUM_POINT_FEATURES);
      for(int i = 0; i < numPoints; i++)
        pointFeatures(input, i, features.data() + (size_t)i * NUM_POINT_FEATURES);
      point.apply(features.data(), numPoints, trunk.data() + c);
    }
    {
      vector<float> g(c);
      global.apply(input.global, 1, g.data());
      for(int t = 0; t < seqLen; t++)
        for(int i = 0; i < c; i++)
          trunk[(size_t)t * c + i] += g[i];
    }

    // Trunk -------------------------------------------------------------------------------------------------------
    RopeTables tables;
    vector<float> tmp((size_t)seqLen * c);
    vector<float> mid((size_t)seqLen * midChannels);
    vector<float> midTmp((size_t)seqLen * midChannels);
    vector<float> attnOut;
    for(const TrunkBlock& block : blocks) {
      block.pre.apply(trunk.data(), seqLen, tmp.data());
      block.preConv.apply(tmp.data(), seqLen, mid.data());
      size_t attentionIdx = 0;
      size_t ffnIdx = 0;
      for(bool isAttention : block.isAttention) {
        if(isAttention) {
          const Attention& layer = block.attentions[attentionIdx++];
          layer.norm.apply(mid.data(), seqLen, midTmp.data());
          tokenTables(layer, input, tables);
          attend(layer, midTmp.data(), seqLen, tables, midTmp.data(), seqLen, tables, attnOut);
          for(size_t i = 0; i < mid.size(); i++)
            mid[i] += attnOut[i];
        }
        else {
          block.ffns[ffnIdx++].applyResidual(mid.data(), seqLen);
        }
      }
      block.post.apply(mid.data(), seqLen, midTmp.data());
      block.postConv.apply(midTmp.data(), seqLen, tmp.data());
      for(size_t i = 0; i < trunk.size(); i++)
        trunk[i] += tmp[i];
    }
    {
      // Spatial RMSNorm over all tokens and channels
      double sumSq = 0.0;
      for(float x : trunk)
        sumSq += (double)x * x;
      const float meanSq = (float)(sumSq / ((double)c * seqLen + tipEpsilon));
      const float inv = 1.0f / sqrtf(meanSq + tipEpsilon);
      for(int t = 0; t < seqLen; t++)
        for(int i = 0; i < c; i++) {
          float& x = trunk[(size_t)t * c + i];
          x = activate(tipActivation, x * inv * tipGamma[i] + tipBeta[i]);
        }
    }

    const float boardOffset = sqrtf(xSize * ySize) - 14.0f;

    // Candidate decoder ---------------------------------------------------------------------------------------------
    const bool hasGround = input.groundAllowed;
    const int numQueries = numCands + (hasGround ? 1 : 0);
    result.numCandidates = numCands;
    result.hasGround = hasGround;
    result.policy.assign((size_t)numQueries * 2, 0.0f);
    if(numQueries > 0) {
      const int cd = decoderChannels;
      vector<float> pooled((size_t)3 * c, 0.0f);
      for(int i = 0; i < c; i++)
        pooled[2 * c + i] = -1e30f;
      for(int t = 0; t < seqLen; t++)
        for(int i = 0; i < c; i++) {
          const float x = trunk[(size_t)t * c + i];
          pooled[i] += x;
          pooled[2 * c + i] = std::max(pooled[2 * c + i], x);
        }
      for(int i = 0; i < c; i++) {
        pooled[i] /= (float)seqLen;
        pooled[c + i] = pooled[i] * (boardOffset / 10.0f);
      }
      vector<float> poolProj(cd);
      pool.apply(pooled.data(), 1, poolProj.data());

      vector<float> queries((size_t)numQueries * cd);
      if(numCands > 0) {
        vector<float> features((size_t)numCands * NUM_CAND_FEATURES);
        for(int i = 0; i < numCands; i++)
          candidateFeatures(input, i, features.data() + (size_t)i * NUM_CAND_FEATURES);
        cand.apply(features.data(), numCands, queries.data());
      }
      if(hasGround) {
        const float groundValue = groundFeature(input);
        ground.apply(&groundValue, 1, queries.data() + (size_t)numCands * cd);
      }
      for(int qi = 0; qi < numQueries; qi++)
        for(int i = 0; i < cd; i++)
          queries[(size_t)qi * cd + i] += poolProj[i];

      vector<float> queriesNorm((size_t)numQueries * cd);
      RopeTables queryTables;
      for(size_t l = 0; l < decoderAttentions.size(); l++) {
        const Attention& layer = decoderAttentions[l];
        layer.norm.apply(queries.data(), numQueries, queriesNorm.data());
        candidateTables(layer, input, numQueries, queryTables);
        tokenTables(layer, input, tables);
        attend(layer, queriesNorm.data(), numQueries, queryTables, trunk.data(), seqLen, tables, attnOut);
        for(size_t i = 0; i < queries.size(); i++)
          queries[i] += attnOut[i];
        decoderFFNs[l].applyResidual(queries.data(), numQueries);
      }
      decoderNorm.apply(queries.data(), numQueries, queriesNorm.data());
      decoderOut.apply(queriesNorm.data(), numQueries, result.policy.data());
    }

    // Value head ------------------------------------------------------------------------------------------------------
    {
      const int cv1 = v1.outC;
      vector<float> outv1((size_t)seqLen * cv1);
      v1.apply(trunk.data(), seqLen, outv1.data());
      activateAll(v1Activation, outv1.data(), outv1.size());
      vector<float> pooled((size_t)3 * cv1, 0.0f);
      for(int t = 0; t < seqLen; t++)
        for(int i = 0; i < cv1; i++)
          pooled[i] += outv1[(size_t)t * cv1 + i];
      for(int i = 0; i < cv1; i++) {
        pooled[i] /= (float)seqLen;
        pooled[cv1 + i] = pooled[i] * (boardOffset / 10.0f);
        pooled[2 * cv1 + i] = pooled[i] * (boardOffset * boardOffset / 100.0f - 0.1f);
      }
      vector<float> outv2(v2.outC);
      v2.apply(pooled.data(), 1, outv2.data());
      activateAll(v2Activation, outv2.data(), outv2.size());
      v3.apply(outv2.data(), 1, result.value);
      sv3.apply(outv2.data(), 1, result.scoreValue);
      result.ownership.assign(numPoints, 0.0f);
      if(numPoints > 0)
        ownership.apply(outv1.data() + cv1, numPoints, result.ownership.data());
    }
  }
};

SparseDotsNN::Model::Model(const SparseDotsDesc& desc) : impl(std::make_unique<Impl>(desc)) {}

SparseDotsNN::Model::~Model() = default;

void SparseDotsNN::Model::evaluate(const DotsSparseInput& input, RawOutput& result) const {
  impl->evaluate(input, result);
}

int SparseDotsNN::effectiveSymmetry(const int symmetry, const int xSize, const int ySize) {
  if(SymmetryHelpers::isTranspose(symmetry) && xSize != ySize)
    return symmetry & 0x3;
  return symmetry;
}

void SparseDotsNN::writeOutput(const DotsSparseInput& input, const RawOutput& raw, const float policyOptimism, NNOutput& output) {
  const int nnXLen = output.nnXLen;
  const int nnYLen = output.nnYLen;
  testAssert((int)input.candidateLocs.size() == raw.numCandidates);
  std::fill(output.policyProbs, output.policyProbs + NNPos::MAX_NN_POLICY_SIZE, 0.0f);
  auto mixed = [&](int i) {
    const float p = raw.policy[(size_t)i * 2];
    const float pOpt = raw.policy[(size_t)i * 2 + 1];
    return p + (pOpt - p) * policyOptimism;
  };
  for(int i = 0; i < raw.numCandidates; i++)
    output.policyProbs[NNPos::locToPos(input.candidateLocs[i], input.xSize, nnXLen, nnYLen)] = mixed(i);
  if(raw.hasGround)
    output.policyProbs[NNPos::getPassPos(nnXLen, nnYLen)] = mixed(raw.numCandidates);

  output.whiteWinProb = raw.value[0];
  output.whiteLossProb = raw.value[1];
  output.whiteNoResultProb = raw.value[2];
  output.whiteScoreMean = raw.scoreValue[0];
  output.whiteScoreMeanSq = raw.scoreValue[1];
  output.whiteLead = raw.scoreValue[2];
  output.varTimeLeft = raw.scoreValue[3];
  output.shorttermWinlossError = raw.scoreValue[4];
  output.shorttermScoreError = raw.scoreValue[5];

  if(output.whiteOwnerMap != NULL) {
    std::fill(output.whiteOwnerMap, output.whiteOwnerMap + nnXLen * nnYLen, 0.0f);
    testAssert(raw.ownership.size() == input.pointLocs.size());
    for(size_t i = 0; i < input.pointLocs.size(); i++)
      output.whiteOwnerMap[NNPos::locToPos(input.pointLocs[i], input.xSize, nnXLen, nnYLen)] = raw.ownership[i];
  }
}
