## Improved Visual Attention Classification for Autism Spectrum Disorder through Time-Dependent Representations.

People with autism spectrum disorder (ASD) tend to attend differently to visual stimuli when compared to neurotypical individuals. These differences in visual attention can be used for the automatic screening of ASD. We classify a given sequence of fixations on a particular image belonging to either a person not having or having ASD. A deep learning framework is proposed which extracts visual feature embeddings around each point of fixation and processes them sequentially using recurrent neural networks (LSTM) [1] or using transformers [2]. We also propose to exploit temporal information of each fixation such as duration using time-masking and time-event joint embedding techniques [3].

With duration-sensitive visual features for each fixation, we achieve an improvement of 11.46% in the F1 score compared to the baseline recurrent neural network without duration encoded into it. We also experiment with transformers and demonstrate their potential for the classification of visual attention.

All experiments are performed on the Saliency4ASD [4] dataset which contains 300 images with around 24 fixations per image where half of them are from people with ASD and the other half from people without ASD. 10% of the data is used for validation and the remaining 90% is used for training. The dataset is available at [link](https://zenodo.org/record/2647418#.YyG2ttJBwz1).

<hr>
There are two notebooks in the repo, which can be run on colab to train LSTM and Transformer networks respectively. The notebooks are self-explanatory and can be run without any changes. The data is downloaded using wget library in the notebook itself.

<hr>
There is also a folder named "Data Preprocessing" which contain notebooks and scripts to preprocess the following datasets:

1. MIT1003 [5]
2. CAT2000 [6]
3. SALICON [7]
4. VIU [8]

## References

1.  [Attention-based Autism Spectrum Disorder Screening with Privileged Modality [Chen et al 2019, ICCV]](https://www.semanticscholar.org/paper/Attention-Based-Autism-Spectrum-Disorder-Screening-Chen-Zhao/45b967283dd8e3732284387b36ed5e38a3aed0ff)

2.  [Attention is all you need [Vaswani et al 2017, NIPS]](https://arxiv.org/abs/1706.03762)

3.  [Time-Dependent Representation for Neural Event Sequence Prediction [Li et al 2017, ArXiv]](https://www.semanticscholar.org/paper/Time-Dependent-Representation-for-Neural-Event-Li-Du/ec7bab52b2220a6cad410dd82b3fbe140d2196f0)

4.  [Saliency4ASD: Challenge, dataset and tools for visual attention modeling for autism spectrum disorder [Gutierrez 2021, Signal Process. Image Commun.]](https://www.semanticscholar.org/paper/Saliency4ASD%3A-Challenge%2C-dataset-and-tools-for-for-Guti%C3%A9rrez-Che/26fea28ae570165e22296e36d11ebf656086d240)

5.  [Learning to Predict where Humans Look [Judd et al 2009, ICCV]](http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html)

6.  [CAT2000: A Large Scale Fixation Dataset for Boosting Saliency Research [Itti et al 2015, CVPR]](http://saliency.mit.edu/cat2000_visualization.html)

7.  [SALICON: Saliency in Context [Jiang et al 2015, CVPR]](http://salicon.net/)
