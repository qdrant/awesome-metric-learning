# awesome-metric-learning
😎 Awesome list about practical Metric Learning and its applications

## Surveys 📖
- [What is Metric Learning?](http://contrib.scikit-learn.org/metric-learn/introduction.html)
> A beginner-friendly starting point for traditional metric learning methods from scikit-learn website. It has proceeding guides for [supervised](http://contrib.scikit-learn.org/metric-learn/supervised.html), [weakly supervised](http://contrib.scikit-learn.org/metric-learn/weakly_supervised.html) and [unsupervised](http://contrib.scikit-learn.org/metric-learn/unsupervised.html) metric learning algorithms in [`metric_learn`](http://contrib.scikit-learn.org/metric-learn/metric_learn.html) package.
- [Deep Metric Learning: A Survey](https://www.mdpi.com/2073-8994/11/9/1066/htm)
> A comprehensive study in which factors such as sampling strategies, distance metrics and network structures are systematically analyzed by comparing the quantitative results of the methods.
- [Deep Metric Learning: A (Long) Survey](https://hav4ik.github.io/articles/deep-metric-learning-survey)
> An intuitive survey of the need for metric learning, old and state-of-the-art approaches, and some real world use cases.
- [A Tutorial on Distance Metric Learning: Mathematical Foundations, Algorithms, Experimental Analysis, Prospects and Challenges (with Appendices on Mathematical Background and Detailed Algorithms Explanation)](https://arxiv.org/abs/1812.05944)
> The long title itself is a good description of this great survey. It is intended for those interested in mathematical foundations of metric learning.


## Applications ⚒️
- [Detic](https://github.com/facebookresearch/Detic)
> Code released for ["Detecting Twenty-thousand Classes using Image-level Supervision"](https://arxiv.org/abs/2201.02605). It is an open-class object detector to detect any label encoded by CLIP without finetuning. See [demo](https://huggingface.co/spaces/akhaliq/Detic).


## Libraries 🧰
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
> Developed on top of the well known [Transformers](https://github.com/huggingface/transformers) library, it provides an easy way to finetune Transformer-based models to obtain sequence-level embeddings.
- [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
> Modular, flexible and extensible library implementing losses, miners, samplers and trainers for metric learning in PyTorch.
- [tensorflow-similarity](https://github.com/tensorflow/similarity)
> A library in the TensorFlow ecosystem with a Keras-like API, providing support for self-supervised contrastive learning and state-of-the-art methods such as SimCLR, SimSian and Barlo Twins.


## Papers 🔬
- [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
> Published by Yann Le Cun et al. (2005), its main focus was on dimensionality reduction. However, the method proposed has excellent properties for metric learning such as preserving neighbourhood relationships and generalization to unseen data, and it has extensive applications with a great number of variations ever since. It is advised that you read [this great post](https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246) to better understand its importance for metric learning.
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
> The paper introduces Triplet Loss, which can be seen as the "ImageNet moment" for deep metric learning. It is still one of the state-of-the-art methods, and has a great number of applications in almost any data modalities.
- [Deep Metric Learning with Angular Loss](https://arxiv.org/abs/1708.01682)
> A novel loss function with better properties than Contrastive and Triplet Loss such as scale invariance, robustness against feature variance, and better convergence.
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
> Although it is originally designed for the face recognition task, this loss function achieves state-of-the-art results in many other metric learning problems with a simpler and faster data feeding. It is also robust against unclean and unbalanced data when modified with sub-centers and a dynamic margin.
- [Learning Distance Metrics from Probabilistic Information](https://cse.buffalo.edu/~lusu/papers/TKDD2020.pdf)
> Working with datasets that contain probabilistic labels instead of deterministic values.
- [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906)
> The paper introduces a method that explicitly avoids the collapse problem in high dimensions with a simple regularization term on the variance of the embeddings along each dimension individually. This new term can be incorporated into other methods for stabilization of the training and performance improvements.
- [On the Unreasonable Effectiveness of Centroids in Image Retrieval](https://arxiv.org/abs/2104.13643)
> The paper proposes to use the mean centroid representation both during training and retrieval for robustness against outliers, and more stable features. It further reduces retrieval time and storage requirements, so it is suitable for production deployments.


## Datasets ℹ️
> Any labelled or unlabelled data can be used for metric learning with an appropriate method chosen. However some datasets are particularly important in the literature for benchmarking or in some other way, and we list them in this section.

