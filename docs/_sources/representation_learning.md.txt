# Representation Learning

## What is representation learning?

In short, it is the technology at the core of the most popular and impactful foundation models like [GPT](https://stanford-cs324.github.io/winter2022/lectures/introduction/) and and [Stable Diffusion](https://poloclub.github.io/diffusion-explainer/)

In [Kevin Murphy’s biblical treatise on machine learning](https://probml.github.io/pml-book/book2.html), Poole and Kornblinth offer this definition:

*“Representation learning is a paradigm for training machine learning models to transform raw inputs into a form that makes it easier to solve new tasks. Unlike supervised learning, where the task is known at training time, representation learning often assumes that we do not know what task we wish to solve ahead of time.”*

In this article, we will explore techniques that have recently been developed for representation learning. But first, we need to address an often overlooked aspect. What do we mean by “raw inputs?”

## Signal measurement

We digitize the natural world by [sampling](https://en.wikipedia.org/wiki/Sampling_(signal_processing)) and [quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) of continuous signals. A large branch of machine learning research focuses on signals that are measured on uniform, rectangular grids. For example:

- Audio: A continuous-time, analog acoustic pressure signal is converted to discrete-time and digital signal by a  [delta-sigma-modulation](https://en.wikipedia.org/wiki/Delta-sigma_modulation) analog-to-digital converter (ADC).
- Images and video: Light intensity is quantized by an [image sensor](https://en.wikipedia.org/wiki/Image_sensor) and light spectrum is quantized by a [Bayer filter](https://en.wikipedia.org/wiki/Bayer_filter). Light is spatially sampled into a rectangular grid of pixels.

The use of uniform, rectangular grids, vastly simplifies machine learning tasks. In particular, it allows us to leverage the extremely powerful **convolution operation.** 

However, many signal types are not amenable to this regular grid structure. For example:

- A [B-mode medical ultrasound](https://en.wikipedia.org/wiki/Medical_ultrasound#Types) measures high frequency acoustic signals using a curvilinear array consisting of 100s of transducers. Unlike conventional audio, measurement and storage of the raw audio signal is impractical. Instead, mixed analog-digital signal processing is used to produce a digital representation of the soft tissue density as a function of depth and angle, which the operator can examine in real-time.
- a [3D laser scanner](https://en.wikipedia.org/wiki/3D_scanning) measures the time-of-flight of optical pulses swept over a range of angles. In the case of handheld scanners, the received optical pulses are calibrated with respect to the motion of the device. The end result is a point-cloud representation of data which does not naturally conform to any type of grid.

## Rate-distortion theory and lossy compression

Because of the high density and precision of our signal measurements, storage and transmission of the raw signal is rarely practical. For example, a standard 1080p video stream is roughly three billion bits *per second* (2.98 Gbps) before compression, roughly ten times the typical capacity of a 5G cellular network (100-400 Mpbs).

[Lossy compression](https://en.wikipedia.org/wiki/Lossy_compression) techniques aim to reduce the data's size without excessively compromising its quality. [Rate-Distortion Theory](https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory), a branch of [Information Theory](https://en.wikipedia.org/wiki/Information_theory) provides a theoretical framework for understanding the trade-off between the amount of data (rate) needed to represent a source and the loss of quality (distortion) of the representation.

While the bit rate is well defined, "distortion" is much trickier. Simple distortion metrics like mean squared error are only loosely correlated to the quality metrics that we actually care about. In audio and video signal processing, a variety of perceptual quality metrics were co-developed with lossy compression algorithms which are now understood to have extremely deep connections with natural signal statistics.

## The holy grail of representation learning

Before discussing specific techniques for representation learning, let us enumerate a wish list of design goals that will be helpful to understand their tradeoffs

An ideal representation learning algorithm would produce an encoder and decoder pair (codec) that:

1. Is computationally efficient
2. Allows graceful control between rate and distortion
3. Uses a training process which is scalable to the size of a foundation model
4. Is applicable to a variety of modalities, and is not limited to grid-structured inputs
5. Simplifies or enables downstream tasks:
    - Analysis (e.g. detection, classification, segmentation)
    - Synthesis (e.g. inpainting, super-resolution, deblurring)
    - Many others, including
        - Search and retrieval
        - Novel view synthesis
        - Semantic editing
        - Source separation

For now, different representations each have their own uses for certain tasks. We don’t need a representation to be perfect at everything for it to be useful, but these are the goals we should keep in mind.

## Comparison of techniques

|  | Efficiency | R-D control | Analysis | Synthesis | Search |
| --- | --- | --- | --- | --- | --- |
| Conventional lossy codecs | ✅ | ✅ | ❌ | ❌ | ❌ |
| Classifier decapitation | ✅ | ❌ | ✅ | ❌ | ✅ |
| $\nabla_x$ embedding |  |  |  |  |  |
| Autoencoders |  |  |  |  |  |
| INRs/functa |  |  |  |  |  |

## Conventional codecs

Codecs such as MPEG and JPEG were standardized in the 1990s and are still widely used for audio, images, and video. Generally, these codecs adhere to the paradigm of [transform coding](https://en.wikipedia.org/wiki/Transform_coding), which involves three main steps:

1. **A time-frequency or space-frequency sub-band decomposition.** For example, JPEG uses the [two-dimensional type-II DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform#Multidimensional_DCTs) applied to 8x8 blocks. In MPEG layer III, an audio signal is efficiently divided into 32 frequency bands using a [polyphase filterbank](https://en.wikipedia.org/wiki/Polyphase_quadrature_filter), then each sub-band is further decomposed using the [modified DCT](https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform). The transform has the effect of [decorrelating](https://en.wikipedia.org/wiki/Decorrelation) signal components to eliminate redundancy.
2. **Perceptual quantization.** While data from an audio ADC or image sensor typically have a precision in the range of 4-16 bits, the sub-band decomposition may be carried out with higher precision (often 32-bit floating point) and then quantized to lower precision. a perceptual model is used to allocated bits to each sub-band. The [quantization matrix in JPEG](https://en.wikipedia.org/wiki/JPEG#Quantization), for example, results in between 2-8 bits for each DCT coefficient. Most of the complexity of standard codecs is in the sub-band bit allocation procedure.
3. **Entropy coding.** The lossy compression performed in steps (1) and (2) is combined with lossless compression to achieve very high compression ratios. Simple prefix codes such as the [Huffman code](https://en.wikipedia.org/wiki/Huffman_coding), [Golomb code](https://en.wikipedia.org/wiki/Golomb_coding), or [run-length encoding](https://en.wikipedia.org/wiki/Run-length_encoding) are usually preferred to keep the decoding cheap.

## Classifier decapitation

This is a technique in which the final layer of a pre-trained classifier is removed, and the semifinal layer is used as an representation. When the classifier has been trained on a large and diverse dataset, this is an effective method to apply transfer learning with a different dataset or a different set of classes by training a sigle fully connected layer. This approach is also the basis for FID and FAD, the standard evaluation metrics for generative models of images and audio respectively.

## Vector embedding

The concept of a vector embedding (which we will henceforth refer to as "$\nabla _x$ embedding” for reasons that will soon become clear) is best explained in the context of natural language processing.

### Example: alphabetic character embedding

Consider an input character sequence from a discrete alphabet (for example the 128 valid ASCII characters). 

The standard training procedure for neural networks consists of a high precision (16 or 32 bit floating point) representation at the input neurons and some form of divisive normalization. This works amazingly well because of the ability of the floating point number system to efficiently represent real numbers.

When training a neural network, how should we represent our 7-bit characters which represent a category rather than a real number? One approach is to simply one-hot encode. This works, but leads to unreasonable memory requirements and dramatically increases the number of network parameters. The “$\nabla _x$ embedding" technique instead consists of the following:

- Create a lookup table that maps the each of the 128 possible input tokens to a unique vector in an $M$ dimensional space. We will call this $128 \times M$ matrix the "embedding matrix" and we will initialize it with i.i.d. samples from a standard Gaussian.
- When training, map the input tokens to vectors according to the embedding matrix/lookup table. Use the result of this mapping as the input to the network instead of the categorical variables.
- Train the neural network as usual. That is, compute the gradient of the loss function with respect to the neural network parameters, $\nabla_\theta L$. Moving in the opposite direction of this gradient will adjust the parameters to minimize the loss.
- In addition, compute the gradient of the loss function with respect to the **input**, $\nabla_x L$. Since the input is simply the entries of the embedding table, moving in the opposite direction will **learn a representation** that minimizes the loss.

## Autoencoders

A plethora of architectures for autoencoders have been proposed. One commonality among them is a **reconstruction objective**: The target output should be similar to the input, meaning that the loss function includes a **distortion metric**. The distortion metric can be chosen for mathematical convenience (e.g. squared error), but for an autoencoder to achieve our design goals it is preferable to choose an application specific metric.

For example, one of the key ingredients enabling [neural image compression](https://www.cns.nyu.edu/~lcv/iclr2017/) is the use [structural similarity (SSIM)](https://en.wikipedia.org/wiki/Structural_similarity) as the distortion metric. In fact, recent research has [established an important link between perceptual quality metrics like SSIM and efficient statistical learning](https://arxiv.org/abs/2106.04427).

Some notable flavors of autoencoders include:

- The **Variational autoencoder (VAE),** which adds an additional loss term that forces the learned representation towards a specific probability distribution (typically Gaussian). Recently, foundation models for [audio](https://audioldm.github.io/), [images](https://en.wikipedia.org/wiki/Stable_Diffusion), and [video](https://research.nvidia.com/labs/toronto-ai/VideoLDM/) have utilized this basic VAE form to improve the efficiency of diffusion models.
- **Vector-quantized VAE (VQ-VAE)**. In addition to the reconstruction loss, a codebook loss similar to the standard vector quantization/k-means objective is added. Neural codecs including [Google Soundstream](https://github.com/wesbz/SoundStream) and [Meta's EnCodec](https://github.com/facebookresearch/encodec) have recently been standardized and use variants of the VQ-VAE to achieve extremely high compression ratios. Recent audio synthesis models for for [speech](https://vall-e.io/) and [music](https://google-research.github.io/seanet/musiclm/examples/) use representations learned from a VQ-VAE.
- The **denoising autoencoder** and **masked autoencoder** leave the target the same, but only provide partial access to the input. This paradigm has been extremely successful in [modeling language](https://en.wikipedia.org/wiki/BERT_(language_model)), [audio](https://arxiv.org/abs/2203.16691), and [images](https://arxiv.org/abs/2111.06377), and is now commonly referred to as “self-supervised” learning
- The **rate-distortion autoencoder**, aka “[nonlinear transform coding](https://ieeexplore.ieee.org/abstract/document/9242247/)” directly penalizes the entropy of the latent representation. When originally proposed for image compression, Balle et al established an important theoretical link between this technique and the VAE.

## Implicit neural representations

An newer technique that is rapidly gaining traction is the concept of implicit neural representations (INRs), now commonly referred to as *functa*. [Neural Networks as Data](https://ora.ox.ac.uk/objects/uuid:c573637c-bf05-4e8a-a8e4-d499eec77446)* provides a comprehensive overview of INRs, and offers an excellent description:

*"Data is often represented by arrays, such as a 2D grid of pixels for images. However, the underlying signal represented by these arrays is often continuous, such as the scene depicted in an image. A powerful continuous alternative to discrete arrays is then to represent such signals with an implicit neural representation (INR), a neural network trained to output the appropriate signal value for any input spatial location. An image for example, can be parameterized by a neural network mapping pixel locations to RGB values."*

## References

1. [Introduction to Large Language Models](https://stanford-cs324.github.io/winter2022/lectures/introduction/)
2. [Diffusion Explainer](https://poloclub.github.io/diffusion-explainer/)
3. [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html)
4. [End-to-end Optimized Image Compression](https://www.cns.nyu.edu/~lcv/iclr2017/)
5. [Reverse Engineering Self-Supervised Learning](https://arxiv.org/abs/2305.15614)
6. [High-Fidelity Image Compression with Score-based Generative Models](https://arxiv.org/abs/2305.18231)
7. [To Compress or Not to Compress- Self-Supervised Learning and Information Theory: A Review](https://arxiv.org/abs/2304.09355)
8. [Bytes Are All You Need: Transformers Operating Directly On File Bytes](https://arxiv.org/abs/2306.00238)
9. [Advances in Foundation Models](https://stanford-cs324.github.io/winter2023/)
10. [On the Opportunities and Risks of Foundation Models](https://crfm.stanford.edu/report.html)
11. [QLoRA: Efficient Finetuning of Quantized LLMs](https://github.com/artidoro/qlora)
12. [SeiT: Storage-efficient Vision Training](https://github.com/naver-ai/seit)
13. [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html)
14. [What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)
15. [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
16. [Diffusion Explainer](https://poloclub.github.io/diffusion-explainer/)
