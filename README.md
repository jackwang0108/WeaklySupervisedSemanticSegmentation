# WeaklySupervisedSemanticSegmentation


## Datasets

### 1. PASCAL VOC 2012

PASCAL VOC 2012 原始数据集的语义分割标注非常少, 只有2913个标注 (见`SegmentationClass`文件夹), 即语义分割样本只有2913个. 其中:

- 1464个样本用于训练集 (见`ImageSets/Segmentation/train.txt`文件)
- 1449个样本用于验证集 (见`ImageSets/Segmentation/val.txt`文件)

因此, 原始的 PASCAL VOC 2012 语义分割标注不足以训练大型的深度学习模型.

在 PASCAL VOC 2012 的基础上, Berkeley 视觉组（Berkeley Vision and Learning Center, BVLC）发布了专注于 语义分割 和 边界检测的 SBD (Semantic Boundaries Dataset) 数据集. 它为 PASCAL VOC 2012 提供了更多更精细的语义分割标注.

因此, 采用的数据集为:

- 训练集: SBD 训练集 + PASCAL VOC 2012 训练集 (去重)
- 验证集: VOC 2012 验证集
