
---
layout:     post
title:      普林斯顿大学ECCV2018:CornerNet-Lite,超越YOLOv3！基于关键点的目标检测
subtitle:   普林斯顿大学ECCV2018:CornerNet-Lite,超越YOLOv3！基于关键点的目标检测
date:       2019-05-03
author:     刘浪
header-img: img/post-bg-debug.png
catalog: true
tags:
    - 计算机视觉
    - 深度学习
    - 机器学习

---

---
layout:     post
title:      定时器 你真的会使用吗？
subtitle:   iOS定时器详解
date:       2016-12-13
author:     BY
header-img: img/post-bg-ios10.jpg
catalog: 	 true
tags:
    - iOS
    - 定时器
---


# 普林斯顿大学ECCV2018:CornerNet-Lite,超越YOLOv3！基于关键点的目标检测

arXiv: https://arxiv.org/abs/1904.08900

github: https://github.com/princeton-vl/CornerNet-Lite

## CornerNet-Lite: Efficient Keypoint Based Object Detection

基于关键点的方法是目标检测中相对较新的范例，消除了对 anchor boxes 的需求并提供了简化的检测框架。基于Keypoint的CornerNet在单级（single-stage）检测器中实现了最先进的精度。然而，这种准确性来自高处理代价。在这项工作中，解决了基于关键点的高效目标检测问题，并引入了CornerNet-Lite。CornerNet-Lite是CornerNet的两种有效变体的组合：CornerNet-Saccade，它使用注意机制消除了对图像的所有像素进行彻底处理的需要，以及引入新的紧凑骨干架构的CornerNet-Squeeze。这两种变体共同解决了有效目标检测中的两个关键用例：在不牺牲精度的情况下提高效率，以及提高实时效率的准确性。CornerNet-Saccade适用于离线处理，将CornerNet的效率提高6.0倍，将COCO的效率提高1.0％。CornerNet-Squeeze适用于实时检测，提高了流行的实时检测器YOLOv3的效率和准确性（在COCO数据集上，CornerNet-Squeeze为34ms / 34.4 mAP，而YOLOv3为39ms / 33mAP）

## 介绍

CornetNet 在 ECCV 2018 （oral）上横空出世，引领基于关键点的目标检测狂潮（最近anchor-free盛行），但这类算法（很多one-stage）也有缺点。虽然mAP有很大提升，但速度上都远不能达到实时

而 本文CornetNet-Lite 是对CornetNet进行优化，提出了两种算法：

+ **CornerNet-Saccade**
+ **CornerNet-Squeeze**


基于关键点的对象检测是一类通过检测和分组关键点来生成对象边界框的方法。其中，**最先进的CornerNet检测并分组边界框的左上角和右下角(判断两个角是否属于一个对象);**它使用一个堆叠的hourglass network来预测拐角的heatmaps，然后使用关联嵌入对它们进行分组。CornerNet允许一个简化的设计，消除了对anchor boxes的需要，并在单级探测器中实现了COCO的最新精度。

论文旨在提高角网的推理效率。任何目标检测器的效率都可以通过两个**正交方向来提高:减少处理的像素数量和减少每个像素的处理量**。探索了这两个方向，并介绍了CornerNet的两种有效变体:CornerNet-Saccade和CornerNet-Squeeze，我们统称为角网CornerNet-Lite

**CornerNet-Squeeze通过减少要处理的像素数来加速inference**。它使用的注意机制类似于人类视觉中的扫视。它从缩小的完整图像开始，生成一个注意力map，然后放大，由模型进一步处理。这与最初的CornerNet的不同之处在于，它是在多个尺度上完全卷积地应用的。通过选择要在高分辨率下检查的subset of crops，CornerNet-Saccade在提高精度的同时提高了速度。COCO上的实验表明，在每幅图像190ms时，CornerNet-Saccade的AP达到43.2%，AP增加1%，比原始CornerNet加速6倍。

**CornerNet-Squeeze通过减少每个像素的处理量来加速推理**。它融合了SqueezeNet和MobileNets的思想，并引入了一种新的紧凑的沙漏骨干，广泛使用1X1个卷积、瓶颈层和深度可分离卷积。使用新的沙漏骨干，CornerNet-Squeeze达到34.4%的AP COCO在30毫秒，同时更准确和更快的YOLOv3(33.0%在39毫秒)。

一个自然的问题是，能否将CornerNet-Squeeze结合起来，进一步提高其效率。实验中给出了一个否定的答案:CornerNet-Squeeze-Saccade比CornerNet-Squeeze更慢，也更不准确。这是因为为了帮助扫视(Saccade)，网络需要能够生成足够精确的注意力地图，但是CornerNet-Squeeze的超紧凑架构没有这种额外的能力。此外，原始的CornerNet应用于多个尺度，这为扫视提供了足够的空间，以减少处理的像素数量。与此相反，由于超紧推理预算，CornerNet-Squeeze已经在单个尺度上应用，这为Saccade节省的空间要小得多。

创新点:总的来说，这两个版本的CornerNet-Lite使基于关键点的方法具有竞争力，涵盖了两个流行的用例:用于离线处理的“CornerNet-Saccade”，在不牺牲准确性的情况下提高效率;用于实时处理的“CornerNet-Squeeze”，在不牺牲效率的情况下提高准确性。

目标检测中的扫视。人类视觉中的扫视是指一系列快速的眼球运动来固定不同的图像区域。在对象检测算法的上下文中，我们广泛地使用这个术语来表示在推理过程中有选择地裁剪和处理图像区域(顺序地或并行地，像素或特征)。

## 方法
CornerNet-Saccade检测图像中可能的目标位置周围的小区域内的目标。它使用缩小的全图像来预测注意力地图和粗边界框;两者都建议可能的对象位置。然后，CornerNet-Saccade通过检查以高分辨率为中心的区域来检测物体。它还可以通过控制每幅图像要处理的目标位置的最大数量来实现精度和效率的交换。pipeline概述如图2所示。接下来，将详细描述每个步骤。

![avatar](/img/img/cornernet1.png)

图2:CornerNet-Saccade的概述。我们预测一组可能的目标位置，从注意地图(attention maps)和边界框生成的缩小后的完整图像。我们放大每个位置，在那个位置周围裁剪一个小区域。然后我们在每个区域检测对象。我们通过对对象位置进行排序，并选择最上面的k个位置来控制处理效率。最后，我们用NMS合并检测

### Estimating Object Locations

CornerNet-Saccade的第一步是获取图像中可能的目标位置。使用缩小的完整图像来预测注意力地图，它同时显示了位置和位置上物体的粗尺度。对于给定的图像，我们通过将图像较长一侧的大小调整到255和192像素，将其缩小到两个尺度。大小192的图像填充0到255，以便并行处理。使用如此低分辨率的图像有两个原因。首先，这一步不应该成为推理时间的瓶颈。其次，网络应该很容易地利用图像中的上下文信息来预测注意图。


对于缩小后的图像，CornerNet-Saccade预测有3个attention maps，一张用于小对象，一张用于中等对象，还有一张用于大对象。如果一个对象的边框长小于32像素，则认为该对象是小的;如果边框长在32到96像素之间，则认为该对象是中等的;如果边框长大于96像素，则认为该对象是大的。分别预测不同对象大小的位置可以让我们更好地控制每个位置的CornerNet-Saccade应该放大多少。我们可以在小对象位置放大更多，而在中等对象位置缩小。

使用不同尺度的特征图来预测注意图。特征图是由CornerNet-Saccade中的backbone network得到的。网络中的每个hourglass模块都应用几个卷积和下采样层来缩小输入特征图的大小。然后通过多个卷积和上采样层将特征映射回原始输入分辨率。利用上采样层的feature maps来预测attention maps。小尺度的特征映射用于较小的对象，粗尺度的特征映射用于较大的对象。通过对每个feature map应用一个3x3的Conv- ReLU模块和一个1x1 Conv-Sigmoid模块来预测注意图。在测试过程中，我们只处理分数高于阈值t的位置，并在实验中设置t = 0.3。

当CornerNetSaccade处理缩小的图像时，它可能检测到图像中的一些对象并为它们生成边界框。从凹凸不平处得到的包装箱化后的图像可能不准确。因此，我们也检查高分辨率的区域，以获得更好的 bounding boxes。在训练过程中，我们将每个包围框在对应的注意力地图上的中心位置设置为正，其余设置为负，然后使用α为2的focal loss

### Detecting Objects

CornerNet-Saccade使用从缩小的图像中获得的位置来确定处理的位置。如果我们从缩小的图像中直接裁剪区域，一些物体可能会变得太小，无法准确检测。因此，应该在第一步获得的尺度信息的基础上，以更高的分辨率检查区域。

对于从attention maps中获得的location，我们可以根据不同物体大小，其缩放尺度不同：  s_{s} 用于小对象， s_{m} 用于中型对象， s_{l} 用于大型对象。s_{s}>s_{m}>s_{l} ，它们分别为4、2、1。在每个可能的位置(x，y)，我们通过 s_{i} 放大缩小图像，其中i∈{s，m，l}取决于粗对象尺度。然后，我们将CornerNetSaccade应用于以位置(x，y)为中心的255×255窗口

从包围框预测中获得的位置提供了关于对象大小的更多信息。我们可以使用边界框的大小来确定缩放比例。缩放范围的确定使得缩放后的边界框的长边对于小对象是24，对于中等对象是64，对于大对象是192。

在检测到可能的目标位置后，我们通过应用soft- nms[2]合并边界框并删除冗余的边界框。当我们对区域进行裁剪时，这些区域可能包括位于作物边界处的对象的部分，如图3所示。检测器可能会为这些对象生成边界框，这些边界框可能不会被软- nms删除，因为它们可能与完整对象的边界框有较低的重叠。因此，我们删除了触及作物边界的边界框。在训练过程中，我们利用 CornerNet中相同的训练损失训练网络来预测角网的热图、嵌入量和偏移量。


总结下思路：

+ 总体思路：对输入图片进行选择性剪裁，对剪裁结果进一步进行物体检测工作。
+ 具体实现：使用小尺寸输入图片，对Backbone中生成三个attention maps，分别对应小/中/大目标，检测目标位置周围小区域的目标；使用大尺寸输入图片，选择第一步提取的部分区域，进行物体检测。
+ 个人理解：将物体检测分为两步，第一步是获取一些原始图片的crop，这些crop中包含物体的可能性大大提高；第二步是以第一步获取的crop结果作为输入，进行物体检测。


### Backbone Network

新的Hourglass由3个module组成，共54层，相比CornerNet里的由两个module组成的104层的Hourglass-104，新的Backbone称为Hourglass-54。

Hourglass-54里的每个module都比104里浅、参数少。下采样的步长为2。每个Hourglass模块会下采样特征图3次，并按(384,384,512)来增加通道数。模块中有一个512通道的残差模块。

### CornerNet-Squeeze

SqueezeNet共使用了3种策略来降低网络复杂度：(1) 使用1×1替换3×3 (2) decreasing input channels to 3×3 kernels (3) 晚一点下采样。

在CornerNet-Squeeze中使用了fire模块替代残差块。受MobileNet的启发，我们把第二层的3×33×3标准卷积替换为3×33×3depth-wise separable 卷积。与CornerNet的对比见表
![avatar](/img/img/2.png)


总结下思路：

CornerNet-Squeeze：

+ 总体思路：输入图片尺寸不变，修改backbone，使得减少计算量、尽可能保持精度。
+ 实现过程不提了，大概就是利用squeezenet mobilenet等小模型思路修改backbone