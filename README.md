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

<details>
<summary><a href="https://arxiv.org/abs/2201.05176">Neural Approaches to Conversational Information Retrieval</a> - A working draft of a 150-page survey book by Microsoft researchers</summary>

</details>


## Applications 🎮

<details>
<summary><a href="https://github.com/openai/CLIP">CLIP</a> - Training a unified vector embedding for image and text. <code>NLP</code> <code>CV</code></summary>

> CLIP offers state-of-the-art zero-shot image classification and image retrieval with a natural language query. See [demo](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb).
</details>

<details>
<summary><a href="https://github.com/descriptinc/lyrebird-wav2clip">Wav2CLIP</a> - Encoding audio into the same vector space as CLIP. <code>Audio</code> </summary>

> This work achieves zero-shot classification and cross-modal audio retrieval from natural language queries.
</details>

<details>
<summary><a href="https://github.com/facebookresearch/Detic">Detic</a> - Code released for <a href="https://arxiv.org/abs/2201.02605">"Detecting Twenty-thousand Classes using Image-level Supervision"</a>. <code>CV</code></summary>

> It is an open-class object detector to detect any label encoded by CLIP without finetuning. See [demo](https://huggingface.co/spaces/akhaliq/Detic).
</details>

<details>
<summary><a href="https://tfhub.dev/google/collections/gtr/1">GTR</a> - Collection of Generalizable T5-based dense Retrievers (GTR) models. <code>NLP</code></summary>

> TensorFlow Hub offers a collection of pretrained models from the paper [Large Dual Encoders Are Generalizable Retrievers](https://arxiv.org/abs/2112.07899).
> GTR models are first initialized from a pre-trained T5 checkpoint. They are then further pre-trained with a set of community question-answer pairs. Finally, they are fine-tuned on the MS Marco dataset.
> The two encoders are shared so the GTR model functions as a single text encoder. The input is variable-length English text and the output is a 768-dimensional vector.
</details>

<details>
<summary><a href="https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md">TARS</a> - Task-aware representation of sentences, a novel method for several zero-shot tasks including NER. <code>NLP</code></summary>

> The method and pretrained models found in Flair go beyond zero-shot sequence classification and offers zero-shot span tagging abilities for tasks such as named entity recognition and part of speech tagging.
</details>

<details>
<summary><a href="https://github.com/MaartenGr/BERTopic">BERTopic</a> - A novel topic modeling toolkit with BERT embeddings. <code>NLP</code></summary>

> It leverages HuggingFace Transformers and c-TF-IDF to create dense clusters allowing for easily interpretable topics while keeping important words in the topic descriptions. It supports guided, (semi-) supervised, and dynamic topic modeling beautiful visualizations.
</details>

<details>
<summary><a href="https://github.com/ma921/XRDidentifier">XRD Identifier</a> - Fingerprinting substances with metric learning</summary>

> Identification of substances based on spectral analysis plays a vital role in forensic science. Similarly, the material identification process is of paramount importance for malfunction reasoning in manufacturing sectors and materials research.
> This models enables to identify materials with deep metric learning applied to X-Ray Diffraction (XRD) spectrum. Read [this post](https://towardsdatascience.com/automatic-spectral-identification-using-deep-metric-learning-with-1d-regnet-and-adacos-8b7fb36f2d5f) for more background.
</details>

<details>
<summary><a href="https://github.com/overwindows/SemanticCodeSearch">Semantic Code Search</a> - Retrieving relevant code snippets given a natural language query. <code>NLP</code></summary>

> Different from typical information retrieval tasks, code search requires to bridge the semantic gap between the programming language and natural language, for better describing intrinsic concepts and semantics. The repository provides the pretrained models and source code for [Learning Deep Semantic Model for Code Search using CodeSearchNet Corpus](https://arxiv.org/abs/2201.11313), where they apply several tricks to achieve this.
</details>

<details>
<summary><a href="https://git.tu-berlin.de/rsim/duch">DUCH: Deep Unsupervised Contrastive Hashing</a> - Large-scale cross-modal text-image retrieval in remote sensing with computer vision. <code>CV</code> <code>NLP</code></summary>

</details>

<details>
<summary><a href="https://github.com/geekinglcq/HRec">DUration: Deep Unsupervised Representation for Heterogeneous Recommendation</a> - Recommending different types of items efficiently. <code>RecSys</code></summary>

> State-of-the-art methods are incapable of leveraging attributes from different types of items and thus suffer from data sparsity problems because it is quite challenging to represent items with different feature spaces jointly. To tackle this problem, they propose a kernel-based neural network, namely deep unified representation (DURation) for heterogeneous recommendation, to jointly model unified representations of heterogeneous items while preserving their original feature space topology structures. See [paper](https://arxiv.org/abs/2201.05861).
</details>

<details>
<summary><a href="https://github.com/MathieuCayssol/Item2Vec">Item2Vec</a> - Word2Vec-inspired model for item recommendation. <code>RecSys</code></summary>

> It provides the implementation of [Item2Vec: Neural Item Embedding for Collaborative Filtering](https://arxiv.org/abs/1603.04259), wrapped as a `sklearn` estimator compatible with `GridSearchCV` and `BayesSearchCV` for hyperparameter tuning.
</details>

## Case Studies ✍️
<details>
<summary><a href="https://arxiv.org/pdf/1810.09591.pdf">Applying Deep Learning to Airbnb Search</a></summary>
</details>

<details>
<summary><a href="https://arxiv.org/pdf/2106.09297.pdf">Embedding-based Product Retrieval in Taobao Search</a>
</details>

<details>
<summary><a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf">Deep Neural Networks for Youtube Recommendations</a></summary>
</details>


## Libraries 🧰

<details>
<summary><a href="https://github.com/UKPLab/sentence-transformers">sentence-transformers</a> - A library for 
sentence-level embeddings. <code>NLP</code></summary>

> Developed on top of the well-known [Transformers](https://github.com/huggingface/transformers) library, it provides an easy way to finetune Transformer-based models to obtain sequence-level embeddings.
</details>

<details>
<summary><a href="https://github.com/NTMC-Community/MatchZoo">MatchZoo</a> - a collection of deep learning models for matching documents. <code>NLP</code></summary>

> The goal of MatchZoo is to provide a high-quality codebase for deep text matching research, such as document retrieval, question answering, conversational response ranking, and paraphrase identification.
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
<summary><a href="https://github.com/explosion/sense2vec">sense2vec</a> - Contextually keyed word vectors. <code>NLP</code></summary>

> A PyTorch library to train and inference with contextually-keyed word vectors augmented with part-of-speech tags to achieve multi-word queries.
</details>

<details>
<summary><a href="https://github.com/lightly-ai/lightly">lightly</a> - A Python library for self-supervised learning on images. <code>CV</code></summary>

> A PyTorch library to efficiently train self-supervised computer vision models with state-of-the-art techniques such as SimCLR, SimSian, Barlow Twins, BYOL, among others.
</details>

<details>
<summary><a href="https://github.com/lyst/lightfm">LightFM</a> - A Python implementation of a number of popular 
recommender algorithms. <code>RecSys</code></summary>

> It supports incorporating user and item features to the traditional matrix factorization. It represents users and items as a sum of the latent representations of their features, thus achieving a better generalization.
</details>

<details>
<summary><a href="https://github.com/RaRe-Technologies/gensim">gensim</a> - Library for topic modelling, document indexing and similarity retrieval with large corpora</summary>

> It provides efficient multicore and memory-independent implementations of popular algorithms, such as online Latent Semantic Analysis (LSA/LSI/SVD), Latent Dirichlet Allocation (LDA), Random Projections (RP), Hierarchical Dirichlet Process (HDP) or word2vec.
</details>

<details>
<summary><a href="https://github.com/AmazingDD/daisyRec">DasyRec</a> - A library for recommender system development in pytorch. <code>RecSys</code></summary>

> It provides implementations of algorithms such as KNN, LFM, SLIM, NeuMF, FM, DeepFM, VAE and so on, in order to ensure fair comparison of recommender system benchmarks.
</details>


## Tools ⚒️

<details>
<summary><a href="https://projector.tensorflow.org/">Embedding Projector</a> - A web-based tool to visualize high-dimensional data.</summary>

> It supports UMAP, T-SNE, PCA, or custom techniques to analyze embeddings of encoders.
</details>

### Approximate Nearest Neighbors ⚡
<details>
<summary><a href="https://github.com/erikbern/ann-benchmarks">ANN Benchmarks</a> - Benchmarking various ANN implementations for different metrics.</summary>

> It provides benchmarking of 20+ ANN algorithms on nine standard datasets with support to bring your dataset. ([Medium Post](https://medium.com/towards-artificial-intelligence/how-to-choose-the-best-nearest-neighbors-algorithm-8d75d42b16ab?sk=889bc0006f5ff773e3a30fa283d91ee7))
</details>

<details>
<summary><a href="https://github.com/facebookresearch/faiss">FAISS</a> - Efficient similarity search and clustering of dense vectors that possibly do not fit in RAM</summary>

> It is not the fastest ANN algorithm but achieves memory efficiency thanks to various quantization and indexing methods such as IVF, PQ, and IVF-PQ. ([Tutorial](https://www.pinecone.io/learn/faiss-tutorial/))
</details>

<details>
<summary><a href="https://github.com/nmslib/hnswlib">HNSW</a> - Hierarchical Navigable Small World graphs</summary>

> It is still one of the fastest ANN algorithms out there, requiring relatively a higher memory usage. (Paper: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320))
</details>

<details>
<summary><a href="https://github.com/google-research/google-research/tree/master/scann">Google's SCANN</a> - The technology behind vector search at Google</summary>

> Paper: [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396)
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

<details>
<summary><a href="http://arxiv.org/abs/2002.05709">SimCLR: A Simple Framework for Contrastive Learning of Visual Representations</a> - Self-Supervised method comparing two differently augmented versions of the same image with Contrastive Loss. <code>CV</code></summary>

> It demonstrates among other things that
> - composition of data augmentations plays a critical role - Random Crop + Random Color distortion provides the best downstream classifier accuracy,
> - introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations,
> - and Contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning.
</details>

<details>
<summary><a href="https://aclanthology.org/2021.emnlp-main.552">SimCSE: Simple Contrastive Learning of Sentence Embeddings</a> - An unsupervised approach, which takes an input sentence and predicts itself in a contrastive objective, with only standard dropout used as noise. <code>NLP</code></summary>

> They also incorporates annotated pairs from natural language inference datasets into their contrastive learning framework in a supervised setting, showing that contrastive learning objective regularizes pre-trained embeddings’ anisotropic space to be more uniform, and it better aligns positive pairs when supervised signals are available.
</details>

<details>
<summary><a href="http://arxiv.org/abs/2103.00020">Learning Transferable Visual Models From Natural Language Supervision</a> - The paper that introduced CLIP: Training a unified vector embedding for image and text. <code>NLP</code> <code>CV</code></summary>
</details>

<details>
<summary><a href="http://arxiv.org/abs/2102.05918">Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision</a> - Google's answer to CLIP: Training a unified vector embedding for image and text but using noisy text instead of a carefully curated dataset. <code>NLP</code> <code>CV</code></summary>
</details>


## Datasets ℹ️
> Practitioners can use any labeled or unlabelled data for metric learning with an appropriate method chosen. However, some datasets are particularly important in the literature for benchmarking or other ways, and we list them in this section.

<details>
<summary><a href="https://nlp.stanford.edu/projects/snli/">SNLI</a> - The Stanford Natural Language Inference Corpus, 
serving as a useful benchmark. <code>NLP</code></summary>

> The dataset contains pairs of sentences labeled as `contradiction`, `entailment`, and `neutral` regarding semantic relationships. Useful to train semantic search models in metric learning.
</details>

<details>
<summary><a href="https://cims.nyu.edu/~sbowman/multinli/">MultiNLI</a> - NLI corpus with samples from multiple genres. <code>NLP</code></summary>

> Modeled on the SNLI corpus, the dataset contains sentence pairs from various genres of spoken and written text, and it also offers a distinctive cross-genre generalization evaluation.
</details>

<details>
<summary><a href="https://www.kaggle.com/c/landmark-recognition-2019">Google Landmark Recognition 2019</a> - Label famous (and no so famous) landmarks from images. <code>CV</code></summary>

> Shared as a part of a Kaggle competition by Google, this dataset is more diverse and thus more interesting than the first version.
</details>

<details>
<summary><a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST</a> - a dataset of Zalando's article images. <code>CV</code></summary>

> The dataset consists of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
</details>

<details>
<summary><a href="https://cvgl.stanford.edu/projects/lifted_struct/">The Stanford Online Products dataset</a> - dataset has 22,634 classes with 120,053 product images. <code>CV</code></summary>

> The dataset is published along with ["Deep Metric Learning via Lifted Structured Feature Embedding"](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16) paper.
</details>

<details>
<summary><a href="https://www.drivendata.org/competitions/79/">MetaAI's 2021 Image Similarity Dataset and Challenge</a> - dataset has 1M Reference image set, 1M Training image set, 50K Dev query image set and 50K Test query image set. <code>CV</code></summary>

> The dataset is published along with ["The 2021 Image Similarity Dataset and Challenge"](http://arxiv.org/abs/2106.09672) paper.
</details>
