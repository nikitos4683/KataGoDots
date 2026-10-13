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
  constexpr int NUM_POINT_FEATURES = 12;
  constexpr int NUM_VERTEX_FEATURES = 24;
  // log1p(boundary length) and the 10 area scalars, concatenated after the vertex mean and max
  constexpr int NUM_AREA_SCALAR_FEATURES = 11;
  constexpr int NUM_CAND_FEATURES = 8;
  void pointFeatures(const DotsSparseInput& input, int i, float* out);
  void vertexFeatures(const DotsSparseInput& input, int i, float* out);
  void areaScalarFeatures(const DotsSparseInput& input, int a, float* out);
  void candidateFeatures(const DotsSparseInput& input, int i, float* out);
  float groundFeature(const DotsSparseInput& input);

  // The symmetry actually applied to the position: transposing symmetries need a square board
  int effectiveSymmetry(int symmetry, int xSize, int ySize);

  // Writes the raw outputs into the NNOutput the way the backends do: logits by board position (only the candidates
  // and the grounding are meaningful, the rest is masked by NNEvaluator), the value and score values as they are,
  // and per-dot ownership logits in whiteOwnerMap (converted by NNEvaluator). `input` gives the board locations.
  void writeOutput(const DotsSparseInput& input, const RawOutput& raw, float policyOptimism, NNOutput& output);
}

#endif  // NEURALNET_SPARSEDOTSNN_H_
