# awesome-metric-learning
😎 Awesome list about practical Metric Learning and its applications

## Motivation 🤓
At Qdrant, we have one goal: make metric learning more practical. This listing is in line with this purpose, and we aim at providing a concise yet useful list of awesomeness around metric learning. It is intended to be inspirational for productivity rather than serve as a full bibliography.

If you find it useful or like it in some other way, you may want to join our Discord server, where we are running a paper reading club on metric learning.

<p align=center>
    <a href="https://discord.gg/tdtYvXjC4h"><img src="https://img.shields.io/badge/Discord-Qdrant-5865F2.svg?logo=discord" alt="Discord"></a>
</p>


## Surveys 📖

<details>
<summary><a href='http://contrib.scikit-learn.org/metric-learn/introduction.html'>What is Metric Learning? </a> - A beginner-friendly starting point for traditional metric learning methods from scikit-learn website.</summary>

> It has proceeding guides for [supervised](http://contrib.scikit-learn.org/metric-learn/supervised.html), [weakly supervised](http://contrib.scikit-learn.org/metric-learn/weakly_supervised.html) and [unsupervised](http://contrib.scikit-learn.org/metric-learn/unsupervised.html) metric learning algorithms in [`metric_learn`](http://contrib.scikit-learn.org/metric-learn/metric_learn.html) package.
</details>

<details>
<summary><a href="https://www.mdpi.com/2073-8994/11/9/1066/htm">Deep Metric Learning: A Survey</a> - A comprehensive 
study for newcomers.</summary>

> Factors such as sampling strategies, distance metrics, and network structures are systematically analyzed by comparing the quantitative results of the methods.
</details>

<details>
<summary><a href="https://hav4ik.github.io/articles/deep-metric-learning-survey">Deep Metric Learning: A (Long) Survey</a> - An intuitive survey of the state-of-the-art.</summary>

> It discusses the need for metric learning, old and state-of-the-art approaches, and some real-world use cases.
</details>

<details>
<summary><a href="https://arxiv.org/abs/1812.05944">A Tutorial on Distance Metric Learning: Mathematical Foundations, Algorithms, Experimental Analysis, Prospects and Challenges (with Appendices on Mathematical Background and Detailed Algorithms Explanation)</a> - Intended for those interested in mathematical foundations of metric learning.</summary>

</details>


## Applications ⚒️

<details>
<summary><a href="https://github.com/openai/CLIP">CLIP</a> - Training a unified vector embedding for image and text.</summary>

> CLIP offers state-of-the-art zero-shot image classification and image retrieval with a natural language query. See [demo](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb).
</details>

<details>
<summary><a href="https://github.com/descriptinc/lyrebird-wav2clip">Wav2CLIP</a> - Encoding audio into the same vector 
space as CLIP.</summary>

> This work achieves zero-shot classification and cross-modal audio retrieval from natural language queries.
</details>

<details>
<summary><a href="https://github.com/facebookresearch/Detic">Detic</a> - Code released for <a href="https://arxiv.org/abs/2201.02605">"Detecting Twenty-thousand Classes using Image-level Supervision"</a>.</summary>

> It is an open-class object detector to detect any label encoded by CLIP without finetuning. See [demo](https://huggingface.co/spaces/akhaliq/Detic).
</details>

<details>
<summary><a href="https://github.com/MaartenGr/BERTopic">BERTopic</a> - A novel topic modeling toolkit with BERT 
embeddings.</summary>

> It leverages HuggingFace Transformers and c-TF-IDF to create dense clusters allowing for easily interpretable topics while keeping important words in the topic descriptions. It supports guided, (semi-) supervised, and dynamic topic modeling beautiful visualizations.
</details>

<details>
<summary><a href="https://projector.tensorflow.org/">Embedding Projector</a> - A web-based tool to visualize 
high-dimensional data.</summary>

> It supports UMAP, T-SNE, PCA, or custom techniques to analyze embeddings of encoders.
</details>


## Libraries 🧰

<details>
<summary><a href="https://github.com/UKPLab/sentence-transformers">sentence-transformers</a> - A library for 
sentence-level embeddings.</summary>

> Developed on top of the well-known [Transformers](https://github.com/huggingface/transformers) library, it provides an easy way to finetune Transformer-based models to obtain sequence-level embeddings.
</details>

<details>
<summary><a href="https://github.com/KevinMusgrave/pytorch-metric-learning">pytorch-metric-learning</a> - A modular library implementing losses, miners, samplers and trainers in PyTorch.</summary>

</details>

<details>
<summary><a href="https://github.com/tensorflow/similarity">tensorflow-similarity</a> - A metric learning library in 
TensorFlow with a Keras-like API.</summary>

> It provides support for self-supervised contrastive learning and state-of-the-art methods such as SimCLR, SimSian, and Barlow Twins.
</details>

<details>
<summary><a href="https://github.com/explosion/sense2vec">sense2vec</a> - Contextually keyed word vectors.</summary>

> A PyTorch library to train and inference with contextually-keyed word vectors augmented with part-of-speech tags to achieve multi-word queries.
</details>

<details>
<summary><a href="https://github.com/lightly-ai/lightly">lightly</a> - A Python library for self-supervised learning on images.</summary>

> A PyTorch library to efficiently train self-supervised computer vision models with state-of-the-art techniques such as SimCLR, SimSian, Barlow Twins, BYOL, among others.
</details>

<details>
<summary><a href="https://github.com/lyst/lightfm">LightFM</a> - A Python implementation of a number of popular 
recommender algorithms.</summary>

> It supports incorporating user and item features to the traditional matrix factorization. It represents users and items as a sum of the latent representations of their features, thus achieving a better generalization.
</details>

## Papers 🔬
<details>
<summary><a href="http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf">Dimensionality Reduction by 
Learning an Invariant Mapping</a> - First appearance of Contrastive Loss.</summary>

> Published by Yann Le Cun et al. (2005), its main focus was on dimensionality reduction. However, the method proposed has excellent properties for metric learning such as preserving neighbourhood relationships and generalization to unseen data, and it has extensive applications with a great number of variations ever since. It is advised that you read [this great post](https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246) to better understand its importance for metric learning.
</details>

<details>
<summary><a href="https://arxiv.org/abs/1503.03832">FaceNet: A Unified Embedding for Face Recognition and Clustering</a> - First appearance of Triplet Loss.</summary>

> The paper introduces Triplet Loss, which can be seen as the "ImageNet moment" for deep metric learning. It is still one of the state-of-the-art methods and has a great number of applications in almost any data modality.
</details>

<details>
<summary><a href="https://arxiv.org/abs/1703.07737">In Defense of the Triplet Loss for Person Re-Identification</a> - It shows that triplet sampling matters and proposes to use batch-hard samples.</summary>

</details>

<details>
<summary><a href="https://arxiv.org/abs/1708.01682">Deep Metric Learning with Angular Loss</a> - A novel loss function 
with better properties.</summary>

> It provides scale invariance, robustness against feature variance, and better convergence than Contrastive and Triplet Loss.
</details>

<details>
<summary><a href="https://arxiv.org/abs/1801.07698">ArcFace: Additive Angular Margin Loss for Deep Face Recognition</a> 
> Supervised metric learning without pairs or triplets.</summary>

> Although it is originally designed for the face recognition task, this loss function achieves state-of-the-art results in many other metric learning problems with a simpler and faster data feeding. It is also robust against unclean and unbalanced data when modified with sub-centers and a dynamic margin.
</details>

<details>
<summary><a href="https://cse.buffalo.edu/~lusu/papers/TKDD2020.pdf">Learning Distance Metrics from Probabilistic Information</a> - Working with datasets that contain probabilistic labels instead of deterministic values.</summary>

</details>

<details>
<summary><a href="https://arxiv.org/abs/2105.04906">VICReg: Variance-Invariance-Covariance Regularization for 
Self-Supervised Learning</a> - Better regularization for high-dimensional embeddings.</summary>

> The paper introduces a method that explicitly avoids the collapse problem in high dimensions with a simple regularization term on the variance of the embeddings along each dimension individually. This new term can be incorporated into other methods to stabilize the training and performance improvements.
</details>

<details>
<summary><a href="https://arxiv.org/abs/2104.13643">On the Unreasonable Effectiveness of Centroids in Image Retrieval</a> - Higher robustness against outliers with better efficiency.</summary>

> The paper proposes using the mean centroid representation during training and retrieval for robustness against outliers and more stable features. It further reduces retrieval time and storage requirements, making it suitable for production deployments.
</details>

<details>
<summary><a href="https://arxiv.org/abs/2104.06979">TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning</a> - A SOTA method to learn domain-specific sentence-level embeddings from unlabelled data.</summary>

</details>


## Datasets ℹ️
> Practitioners can use any labeled or unlabelled data for metric learning with an appropriate method chosen. However, some datasets are particularly important in the literature for benchmarking or other ways, and we list them in this section.

<details>
<summary><a href="https://nlp.stanford.edu/projects/snli/">SNLI</a> - The Stanford Natural Language Inference Corpus, 
serving as a useful benchmark.</summary>

> The dataset contains pairs of sentences labeled as `contradiction`, `entailment`, and `neutral` regarding semantic relationships. Useful to train semantic search models in metric learning.
</details>

<details>
<summary><a href="https://cims.nyu.edu/~sbowman/multinli/">MultiNLI</a> - NLI corpus with samples from multiple genres.</summary>

> Modeled on the SNLI corpus, the dataset contains sentence pairs from various genres of spoken and written text, and it also offers a distinctive cross-genre generalization evaluation.
</details>

<details>
<summary><a href="https://www.kaggle.com/c/landmark-recognition-2019">Google Landmark Recognition 2019</a> - Label famous (and no so famous) landmarks from images.</summary>

> Shared as a part of a Kaggle competition by Google, this dataset is more diverse and thus more interesting than the first version.
</details>
