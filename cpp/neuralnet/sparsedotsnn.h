#ifndef NEURALNET_SPARSEDOTSNN_H_
#define NEURALNET_SPARSEDOTSNN_H_

// CPU reference of the sparse Dots models (model version 18), see docs/DotsSparseNN.md.
// Evaluates one position at a time without any padding, in FP32, with the same formulas as
// python/katago/train/model_sparse_dots.py. Used by the Eigen backend and as the reference
// for the GPU backends in the tests.

#include <memory>
#include <vector>

#include "../neuralnet/desc.h"
#include "../neuralnet/nninputs.h"

namespace SparseDotsNN {
  // Raw outputs of a position, from the perspective of the player to move, before the postprocessing
  struct RawOutput {
    int numCandidates = 0;
    bool hasGround = false;
    // (numCandidates + hasGround) x 2: policy and short-term optimistic policy logits, grounding last
    std::vector<float> policy;
    float value[3] = {};
    // score mean, score stdev, lead, variance time, short-term value error, short-term score error (pre-activation)
    float scoreValue[6] = {};
    // Logit per point: the opponent of the owner of the dot captures it by the end of the game
    std::vector<float> ownership;
  };

  class Model {
   public:
    explicit Model(const SparseDotsDesc& desc);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // The input must already have the symmetry applied (DotsSparseInput::withSymmetry)
    void evaluate(const DotsSparseInput& input, RawOutput& result) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl;
  };

  // Input features of the tokens, shared with the GPU backends
  constexpr int NUM_POINT_FEATURES = 19;
  constexpr int NUM_CAND_FEATURES = 8;
  void pointFeatures(const DotsSparseInput& input, int i, float* out);
  void candidateFeatures(const DotsSparseInput& input, int i, float* out);
  float groundFeature(const DotsSparseInput& input);
  // The same with the coordinates and the board size given explicitly, e.g. after a symmetry
  void pointFeaturesAt(const DotsSparseInput::Point& p, int x, int y, int xSize, int ySize, float* out);
  void candidateFeaturesAt(const DotsSparseInput::Candidate& cnd, int x, int y, int xSize, int ySize, float* out);

  // Chain code of a trunk token for the chain attention bias: -1 for GLOBAL, 2 * chain + owner for a point.
  // Equal codes = the same chain, equal parities = the same owner.
  inline int pointChainCode(const DotsSparseInput::Point& p) { return 2 * p.chain + p.owner; }
  // Category of a pair of token codes (index into the chain bias of a head), -1 = no bias (a GLOBAL token)
  inline int chainCategory(int codeA, int codeB) {
    if(codeA < 0 || codeB < 0)
      return -1;
    if(codeA == codeB)
      return 0;
    return (codeA & 1) == (codeB & 1) ? 1 : 2;
  }

  // The symmetry actually applied to the position: transposing symmetries need a square board
  int effectiveSymmetry(int symmetry, int xSize, int ySize);

  // Writes the raw outputs into the NNOutput the way the backends do: logits by board position (only the candidates
  // and the grounding are meaningful, the rest is masked by NNEvaluator), the value and score values as they are,
  // and per-dot ownership logits in whiteOwnerMap (converted by NNEvaluator). `input` gives the board locations.
  void writeOutput(const DotsSparseInput& input, const RawOutput& raw, float policyOptimism, NNOutput& output);
}

#endif  // NEURALNET_SPARSEDOTSNN_H_
