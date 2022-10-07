# Computer Vision
This part is mainly to reproduce various models, in order to better understand them

### 1.Classification
- Vision Transformer
- ResNet:
    - ResNet18, ResNet34, ResNet50, ResNet101, ResNet154:[Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
    - DenseNet121, DenseNet169, DenseNet201, DenseNet264:[Densely connected convolutional networks](https://arxiv.org/pdf/2101.03697.pdf) 
    - ResNeXt:[Aggregated Residual Transformations for Deep Neural Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html) (TODO)
- VGG:
    - VGG16, VGG19:[Very deep convolutional networks for large-scale image recognition](https://arxiv.org/abs/1409.1556)
    - RepVGG_A0-A2, RepVGG_B0:[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf) 
- MobileNet (TODO):
    - MobileNet:[Mobilenets: Efficient convolutional neural networks for mobile vision applications](https://arxiv.org/abs/1704.04861)
    - EfficientNet:[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
    - EfficientNetV2:[EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- Text Recognize: 
    The context-aware model will not be reproduced in the short term
    - CRNN:[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
    - SRN:[Towards Accurate Scene Text Recognition with Semantic Reasoning Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.pdf)
    - DPAN:[Look Back Again: Dual Parallel Attention Network for Accurate and Robust Scene Text Recognition](https://dl.acm.org/doi/pdf/10.1145/3460426.3463674)
    - PARSeq:[Scene Text Recognition withPermuted Autoregressive Sequence Models](https://arxiv.org/pdf/2207.06966.pdf) (Context-Aware Model)
    
- Key Information Extract
    - SDMGR:[Spatial Dual-Modality Graph Reasoning for Key Information Extraction](https://arxiv.org/abs/2103.14470) (TODO)

- Face Recognition
    - DeepFace:[DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://openaccess.thecvf.com/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf)
    - DeepID:[Deep Learning Face Representation from Predicting 10,000 Classes](https://openaccess.thecvf.com/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf)
### 2.Object Detection
- RCNN(TODO):
    - R-CNN:[Rich feature hierarchies for accurate object detection and semantic segmentation](https://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) 
    - R-FCN:[R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://proceedings.neurips.cc/paper/2016/hash/577ef1154f3240ad5b9b413aa7346a1e-Abstract.html)
    - Fast R-CNN:[Fast R-CNN](https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
    - Faster R-CNN:[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html)
- Yolo: 
    - YoloV1:[You Only Look Once:Unified, Real-Time Object Detection](http://arxiv.org/abs/1506.02640)
    - YoloV2:[YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
    - YoloV3:[YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)
    - YoloV4:[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
    - YoloV5: No Paper, https://github.com/ultralytics/yolov5
    - YoloV7:[YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](http://arxiv.org/abs/2207.02696)
- Text Detection:
    - EAST:[EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/pdf/1704.03155v2.pdf)
    - TextBoxes++:[A Single-Shot Oriented Scene Text Detector](https://arxiv.org/pdf/1801.02765.pdf) 
- Detetion Block:
    - SPP:[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://ieeexplore.ieee.org/abstract/document/7005506)
    - FPN:[Feature Pyramid Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)
### 3.Segementation
- Text Detection:
    - PSENet:[Shape Robust Text Detection With Progressive Scale Expansion Network](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)
    - PANNet:[Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network](https://arxiv.org/pdf/1908.05900.pdf)
    - DBNet:[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)
    - DBNet++:[Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/pdf/2202.10304.pdf)
- FCN(TODO)
    - FCN:[Fully Convolutional Networks for Semantic Segmentation](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)
    - SegNet:[Segnet: A deep convolutional encoder-decoder architecture for image segmentation](https://ieeexplore.ieee.org/abstract/document/7803544)
    - U-Net:[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf)
### Other TODO list:
- Process and postprocess module of Yolo
- Postprocess module fo PSENet
- STN module of SRN