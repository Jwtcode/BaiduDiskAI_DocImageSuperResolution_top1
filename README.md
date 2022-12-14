# 百度网盘AI大赛-文档图像超分第1名方案


# 一、赛题分析
此次大赛主题结合日常生活常见情景展开，大家平时在使用手机进行拍照、扫描的时候，往往会需要将图片放大的场景，可是分辨率放大，图片中的元素都变得模糊起来，进而难以使用。本次比赛希望选手们通过算法等知识技术，帮助人们将因为放大而模糊的图片复原，提高图片分辨率，实现文档图像的“无损放大”，让其重新发挥作用。

# 二、 数据分析
- 本次比赛最新发布的数据集共包含训练集、A榜测试集、B榜测试集三个部分，其中训练集共3000个样本，A榜测试集共200个样本，B榜测试集共200个样本,抽取一部分数据如图：
![image](https://github.com/Jwtcode/BaiduDiskAI_DocImageSuperResolution_top1/blob/main/illustration/x.png)
![image](https://github.com/Jwtcode/BaiduDiskAI_DocImageSuperResolution_top1/blob/main/illustration/x2.png)
![image](https://github.com/Jwtcode/BaiduDiskAI_DocImageSuperResolution_top1/blob/main/illustration/x4.png)
- 以上图片分别为原图，放大2倍的标签图片和放大4倍的标签图片。

# 三、模型设计
- 针对图像超分这个任务，我们查阅了相关资料，传统的文本评价标准是以OCR模型识别的准确率来比较的，目的是提高文字识别准确率，但是此次百度网盘的文档超分比赛，评分的标准是PSNR与MS_SSIM，所以该任务属于重建优化的范畴，所以我们选择了基于卷积架构的rcan作为我们此次的baseline,相比于transformer架构的模型，卷积架构会更加稳定。本次比赛要产生原图放大2倍和放大4倍的结果，如果分开计算必然耗时严重，所以我们将两个任务合并。

![image](https://github.com/Jwtcode/BaiduDiskAI_DocImageSuperResolution_top1/blob/main/illustration/pipeline.png)

- 从网络结构图上可以直观的看出改进后rcan由单分支网络变成了双分支网络，一个分支负责产生放大2倍的结果，一个分支负责产生放大4倍的结果。原始的racn使用了均值漂移的聚类算法，考虑到本次训练数据集的像素值大部分只分布在0和255附近，便删去这一部分,不仅如此原始的rcan网络的n_resgroups和n_resblocks分别为20和10，考虑到时效性，我们设置n_resgroups和n_resblocks分别为10和5。

- 训练数据集的内容分为中文图片和英文图片，考虑到语言之间的差异性以及语言文字的先验性，决定中英文分开训练，所以在测试的时候需要一个二分类模型来判断中文还是英文，我们使用轻量化后的的vgg来做二分类。

![image](https://github.com/Jwtcode/BaiduDiskAI_DocImageSuperResolution_top1/blob/main/illustration/vgg.png)

- 由于分类任务非常容易，所以该网络只由几层卷积和非常少的参数量构成。



# 四、数据处理与增强

### 数据划分
- 训练的文档的图像尺寸很大，而且四周存在很多纯色空白，为减小训练过程中的IO占用以及空白样本过于简单不利于模型收敛，应该将其裁剪掉四周的空白区域，在进行切块处理
- 以步长64，大小为128进行切块。

### 数据筛选
- 在进行切块处理后，仍然存在一些图像块是白色的，由于他们很容易恢复，对模型的收敛可能会产生负面的影响，导致收敛不充分。
- 我们通过计算图像梯度，并将图像梯度小于5的图像块删掉。

### 数据增广
- 为保留文字信息的先验性，不进行任何数据增强，只做归一化处理。

# 五、训练细节
- 训练配置
总迭代数：800000 iteration。
我们采用batch size为16来进行训练800000次迭代。
我们采用了余弦退火的学习率策略来优化网络，学习率我们采用1e-4，优化器为Adam。
- 损失函数为L1Loss

# 六、测试细节
- 测试图片预先全部输入到二分类模型中，得到包含中文图片路径的list和包含英文图片路径的list。
- 中文图片路径的list送入到中文模型中，英文图片路径的list送入到英文模型中。
- 在测试中我们输入到模型中的尺寸为256X256,由于我们将原图裁剪，所以图片边缘的处理尤为重要，我们采用overlap的方法，使图片patches之间的重叠部分为24，这就保证了在图片patches复位的过程中前一个patch取到重叠区域的一半，也就是12个像素长度，后一个patch从重叠区域的一半开始取，这就保证了patches边缘在复位的时候可以很好的过渡。
- 整个推理过程为串行.