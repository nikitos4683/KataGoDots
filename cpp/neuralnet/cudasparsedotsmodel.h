#ifndef NEURALNET_CUDASPARSEDOTSMODEL_H_
#define NEURALNET_CUDASPARSEDOTSMODEL_H_

// CUDA implementation of the sparse Dots models (model version 18), see docs/DotsSparseNN.md.
// Evaluates a batch of positions padded to a common token count. The trunk and the candidate decoder run in FP16
// or FP32, the embeddings and the value head always in FP32. Computes the same RawOutput as SparseDotsNN::Model.

#include <memory>
#include <vector>

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/sparsedotsnn.h"

class Logger;

class SparseDotsCudaModel {
 public:
  // The stream and the cublas handle (bound to the stream) are owned by the caller
  SparseDotsCudaModel(
    const SparseDotsDesc& desc, bool useFP16, bool useMmaAttention, cudaStream_t stream, cublasHandle_t cublas, Logger* logger);
  ~SparseDotsCudaModel();
  SparseDotsCudaModel(const SparseDotsCudaModel&) = delete;
  SparseDotsCudaModel& operator=(const SparseDotsCudaModel&) = delete;

  // The inputs must already have their symmetries applied. Synchronizes the stream before returning.
  void evaluate(const std::vector<const DotsSparseInput*>& inputs, std::vector<SparseDotsNN::RawOutput>& outputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

#endif  // NEURALNET_CUDASPARSEDOTSMODEL_H_
