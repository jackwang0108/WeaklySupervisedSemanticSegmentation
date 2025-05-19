# WeaklySupervisedSemanticSegmentation

## Datasets

本项目使用了两个数据集: PASCAL VOC 2012 和 COCO 2014. 这两个数据集本身包含多种类型的标注, 可以用于图像分类/语义分割/关键点分割等任务. 本项目只使用图像和对应的语义分割标注.

### 1. 数据集下载

运行`scripts/download_dataset.sh`脚本, 下载数据集. 该脚本会自动下载数据集并解压到`data/`目录下.

```bash
cd <root-to-project>
bash scripts/download_dataset.sh
```

### 2. 数据集介绍

#### 1. PASCAL VOC 2012

PASCAL VOC 2012 原始数据集的语义分割标注非常少, 只有2913个标注 (见`SegmentationClass`文件夹), 即语义分割样本只有2913个. 其中:

- 1464个样本用于训练集 (见`ImageSets/Segmentation/train.txt`文件)
- 1449个样本用于验证集 (见`ImageSets/Segmentation/val.txt`文件)

因此, 原始的 PASCAL VOC 2012 语义分割标注不足以训练大型的深度学习模型.

在 PASCAL VOC 2012 的基础上, Berkeley 视觉组（Berkeley Vision and Learning Center, BVLC）发布了专注于 语义分割 和 边界检测的 SBD (Semantic Boundaries Dataset) 数据集. 它为 PASCAL VOC 2012 提供了更多更精细的语义分割标注.

因此, 采用的数据集为:

- 训练集: SBD 训练集 + PASCAL VOC 2012 训练集 (去重)
- 验证集: VOC 2012 验证集

#### 2. COCO 2014

原始的 COCO 2014 数据集提供的分割任务标注是实例分割数据集. 其标注形式为 1-N 的标注, 示例如下:

```json
{
    // 假设一张图像（id=1）中有 2 只猫和 1 只狗：
    "images": [{"id": 1, "file_name": "cats_dog.jpg", "width": 800, "height": 600}],
    "annotations": [
        {
        "id": 1, "image_id": 1, "category_id": 1,  // 第一只猫
        "bbox": [100, 200, 150, 150],
        "segmentation": [[110, 210, 120, 220, ...]]
        },
        {
        "id": 2, "image_id": 1, "category_id": 1,  // 第二只猫
        "bbox": [300, 400, 120, 120],
        "segmentation": [[310, 410, 320, 420, ...]]
        },
        {
        "id": 3, "image_id": 1, "category_id": 2,  // 一只狗
        "bbox": [500, 100, 200, 180],
        "segmentation": [[510, 110, 520, 120, ...]]
        }
    ],
    "categories": [
        {"supercategory": "animal", "id": 1, "name": "cat"},
        {"supercategory": "animal", "id": 2, "name": "dog"}
    ]
}

因此, 读取COCO数据集的时候需要从Json标注中生成语义分割标注.

使用的COCO数据集中:

- 训练集: 包含 82081 张 COCO 训练集中的图像
- 验证集: 包含 40137 张 COCO 验证集中的图像
