#ifndef NEURALNET_MODELVERSION_H_
#define NEURALNET_MODELVERSION_H_

// Model versions
namespace NNModelVersion {

  constexpr int latestModelVersionImplemented = 18;
  constexpr int latestInputsVersionImplemented = 8;
  constexpr int defaultModelVersion = 17;

  constexpr int oldestModelVersionImplemented = 3;
  constexpr int oldestInputsVersionImplemented = 3;

  // Dots models use only the sparse token format (see docs/DotsSparseNN.md).
  // The dense V7 Dots models (versions up to 17) are no longer supported.
  constexpr int dotsSparseModelVersion = 18;
  constexpr int dotsSparseInputsVersion = 8;

  // The model version to assume when there is no model file (random/neural-net-less evaluation).
  int getDefaultModelVersion(bool dotsGame);

  // Is this a model in the sparse Dots token format?
  bool isDotsSparse(int modelVersion, bool dotsGame);

  // Throws if the model version cannot be used for the given game.
  void checkSupported(int modelVersion, bool dotsGame);

  // Which V* feature version from NNInputs does a given model version consume?
  int getInputsVersion(int modelVersion, bool dotsGame);

  // Convenience functions, feeds forward the number of features and the size of
  // the row vector that the net takes as input.
  // For the sparse Dots format there are no spatial features and the global features are DotsSparse ones.
  int getNumSpatialFeatures(int modelVersion, bool dotsGame);
  int getNumGlobalFeatures(int modelVersion, bool dotsGame);

  // SGF metadata encoder input versions
  int getNumInputMetaChannels(int metaEncoderVersion);

}  // namespace NNModelVersion

#endif  // NEURALNET_MODELVERSION_H_
