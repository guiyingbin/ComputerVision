# Computer Vision
This part is mainly to reproduce various models, in order to better understand them

### 1.Classification
- Vision Transformer
- ResNet:
    - ResNet18, ResNet34, ResNet50, ResNet101, ResNet154:[Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- Text Recognize:
    - CRNN:[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
    - DPAN:[Look Back Again: Dual Parallel Attention Network for Accurate and Robust Scene Text Recognition](https://dl.acm.org/doi/pdf/10.1145/3460426.3463674) (TODO)
    - PARSeq:[Scene Text Recognition withPermuted Autoregressive Sequence Models](https://arxiv.org/pdf/2207.06966.pdf) (TODO)
    
### 2.Object Detection
- Yolo: 
    - YoloV1:[You Only Look Once:Unified, Real-Time Object Detection](http://arxiv.org/abs/1506.02640)
    - YoloV2:[YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
    - YoloV3:[YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)
    - YoloV4:[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
    - YoloV5: No Paper, https://github.com/ultralytics/yolov5
    - YoloV7:[YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](http://arxiv.org/abs/2207.02696)

### 3.Segementation
- Text Detection:
    - PSENet:[Shape Robust Text Detection With Progressive Scale Expansion Network](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)
    - PANNet:[Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network](https://arxiv.org/pdf/1908.05900.pdf)
    - DBNet:[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)
    - DBNet++:[Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/pdf/2202.10304.pdf)

### TODO:
- Process and postprocess module of Yolo
- Postprocess module fo PSENet