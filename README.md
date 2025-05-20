# Weakly Supervised Semantic Segmentation

**本仓库是一个关注弱监督语义分割的深度学习算法库.**

鉴于**部分弱监督语义分割文章的官方代码无法运行**, **不同的文章适用的预训练数据集不同**等诸多当下弱监督语义分割研究中所存在的问题, 因此, 为了**公平的比较不同的弱监督语义分割方法**, 从而更好的理解弱监督语义分割的研究进展和方法, 本仓库复现了一系列弱监督语义分割方法, 并**在相同的标准下进行测试**.

## TODO LIST

- [x] 复现自己的GradCAM, 不使用pytorch-grad-cam
- [x] 复现PSA

## 复现方法

1. PSA
    - Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation
    - arXiv: [https://arxiv.org/abs/1803.10464](https://arxiv.org/abs/1803.10464)
    - Official Code: [https://github.com/jiwoon-ahn/psa](https://github.com/jiwoon-ahn/psa)
2. 1Stage:
    - Weakly Supervised Semantic Segmentation with One-Stage Object Detection
    - arXiv: [https://arxiv.org/abs/2005.08104](https://arxiv.org/abs/2005.08104)
    - Official Code: [https://github.com/visinf/1-stage-wseg](https://github.com/visinf/1-stage-wseg)
3. AFA
    - Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers
    - arXiv: [https://arxiv.org/abs/2203.16876](https://arxiv.org/abs/2203.16876)
    - Official Code: [https://github.com/rulixiang/afa](https://github.com/rulixiang/afa)
4. CLIP-ES
    - CLIP is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation
    - arXiv: [https://arxiv.org/abs/2303.13969](https://arxiv.org/abs/2303.13969)
    - Official Code: [https://github.com/linyq2117/CLIP-ES](https://github.com/linyq2117/CLIP-ES)
5. WeCLIP
    - Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation
    - arXiv: [https://arxiv.org/abs/2406.11189](https://arxiv.org/abs/2406.11189)
    - Official Code: [https://github.com/zbf1991/WeCLIP](https://github.com/zbf1991/WeCLIP)
6. ExCEL:
    - Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation
    - arXiv: [https://arxiv.org/abs/2406.11189](https://arxiv.org/abs/2406.11189)
    - Official Code: [https://github.com/zwyang6/ExCEL](https://github.com/zwyang6/ExCEL)
