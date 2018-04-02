# X-Detector
X-Detector is a collection of several object detection algorithms. And some of those have not appeared in any academic papers.

Up to now, this repository contains code of the re-implementent of [Light-Head R-CNN](https://arxiv.org/abs/1711.07264) and the second trained version had got to ~70%mAP, more debugging is now in process. 

Besides several other detectors(named X-Det now) are also included, the main idea behind X-Det is to introduce explicit attention mechanisms between feature map channels, so I would like to change its name to "ABCD (**A**ttention **B**etween **C**hannels **D**etector)" later when the performance get to 0.7+mAP on PASCAL VOC 2007 Test Dataset (now only ~0.68mAP was achieved, improvement is still in process).

The pre-trained weight of backbone network can be found in [Resnet-50 backbone](https://github.com/tensorflow/models/tree/master/official/resnet) and [Xception backbone](https://github.com/HiKapok/Xception_Tensorflow). The latest version of PsRoIAlign is [here](https://github.com/HiKapok/PSROIAlign).

You can use part of these codes for your research purpose, but the ideas like the current implementent of X-Det is not allowed to copy without permissions. While the codes for Light-Head R-CNN can be used for your research without any permission but following [Apache License 2.0](https://github.com/HiKapok/X-Detector/blob/master/LICENSE).

Here are some demo result images of X-Det V2(68%mAP), debugging is still in process to make better performance:

![](demo/1.jpg "Detection Example 1")
![](demo/2.jpg "Detection Example 2")
![](demo/3.jpg "Detection Example 3")

## ##
Apache License 2.0
