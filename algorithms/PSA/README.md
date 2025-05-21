# PSA 算法

PSA 算法是对文章 [Learning Pixel-level Semantic Affinity with Image-level Supervision](https://arxiv.org/abs/1803.10464) 的复现

PSA 算法是一个三阶段的算法, 需要训练三个模型:

- 一个 Multi-Label Classification 模型, 用于对整个图像进行多分类
- 一个 AffinityNet, 用于预测像素级别的语义相似度
- 一个 SegmentationNet, 用于对图像进行分割

具体流程如下:

1. 训练 Multi-Label Classification 模型, 用于对整个图像进行多分类
2. 使用训练好的 Multi-Label Classification 模型对图像进行预测, 而后计算 CAM 激活图像
3. 对 CAM 激活图像进行 Refinement, PSA 中使用的是 dCRF
4. 从 Refined CAM 激活图像中计算 Affinity Label, 训练 AffinityNet
5. 使用训练好的 AffinityNet 对图像进行预测, 得到 Affinity Map
6. 使用 Affinity Map revise CAM 激活图像, 最终得到 Pseudo Label
7. 使用最终的 Pseudo Label 训练 SegmentationNet
