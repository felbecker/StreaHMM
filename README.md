# <img src="./misc/streaHMM_logo.png" width="100"> hmmble

A framework for combining hidden Markov models (HMMs) (and related models like conditional random fields (CRFs)) with modern deep learning.

Targeted features:

- Implements algorithms for gradient-based HMM training and inference.
- Modular emission distributions (discrete, continuous).
- Supports multiple, parallel HMMs with variable architectures.
- Implements parallel variants of all algorithms to support ultra-long sequences.
- Can be used with multiple machine learning frameworks: PyTorch and TensorFlow.

