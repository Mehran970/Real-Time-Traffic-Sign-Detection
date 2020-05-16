# Real-Time-Traffic-Sign-Detection
Traffic sign detection by Tensorflow object detection


## Requirements
This project is implemented in [Tensorflow 2](https://www.tensorflow.org/) and it is based on  [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

For install Tensorflow 2, just use this command:
```
pip install tensorflow==2.0.0 
```

## Pretrained models
You can find models from following links and put them inside "models" directory:
- [faster_rcnn_inception_resnet_v2_atrous](https://drive.google.com/open?id=12vLvA9wyJ9lRuDl9H9Tls0z5jsX0I0Da)
- [faster_rcnn_inception_v2](https://drive.google.com/open?id=1LRCSWIkX_i6ijScMfaxSte_5a_x9tjWF)
- [faster_rcnn_resnet_101](https://drive.google.com/open?id=15OxyPlqyOOlUdsbUmdrexKLpHy1l5tP9)
- [faster_rcnn_resnet50](https://drive.google.com/open?id=1aEqlozB_CzhyJX_PO6SSiM-Yiv3fuO8V)
- [rfcn_resnet101](https://drive.google.com/open?id=1eWCDZ5BxcEa7n_jZmWUr2kwHPBi5-SMG)
- [ssd_inception_v2](https://drive.google.com/open?id=1TKMd-wIZJ1aUcOhWburm2b6WgYnP0ZK6)
- [ssd_mobilenet_v1](https://drive.google.com/open?id=1U31RhUvE1Urr5Q92AJynMvl-oFBVRxxg)

for example:
```
├── models
│   ├── faster_rcnn_inception_resnet_v2_atrous
│   ├── ssd_mobilenet_v1
```
Note modify default model name in line 34 of code as name of model in models directory.

## Results
Acccording to **[Evaluation of deep neural networks for traffic sign detection systems](https://doi.org/10.1016/j.neucom.2018.08.009)**
paper:
Faster R-CNN Inception Resnet V2 model obtains the best mAP, while R-FCN Resnet 101 strikes the best trade-off between accuracy and execution time. SSD Mobilenet  is the fastest and the lightest model in terms of memory consumption, making it an optimal choice for deployment in mobile and embedded devices.

| model                            | mAP   | parameters | flops         | memory_mb    | total_exec_millis | accelerator_exec_millis | cpu_exec_millis |
|----------------------------------|-------|------------|---------------|--------------|-------------------|-------------------------|-----------------|
| Faster R-CNN Resnet 50           | 91.52 | 43337242   | 533575386662  | 5256.454615  | 104.0363553       | 75.93395395             | 28.10240132     |
| Faster R-CNN Resnet 101          | 95.08 | 62381593   | 625779295782  | 6134.705805  | 123.2729175       | 90.33714433             | 32.9357732      |
| Faster R-CNN Inception V2        | 90.62 | 12891249   | 120621363525  | 2175.206857  | 58.53338971       | 38.76813971             | 19.76525        |
| Faster R-CNN Inception Resnet V2 | 95.77 | 59412281   | 1837544257834 | 18250.446008 | 442.2206796       | 366.1586796             | 76062           |
| R-FCN Resnet 101                 | 95.15 | 64594585   | 269898731281  | 3509.75153   | 85.45207971       | 52.40321739             | 33.04886232     |
| SSD Mobilenet                    | 61.64 | 5572809    | 2300721483    | 94.696119    | 15.14525          | 4.021267857             | 11.12398214     |
| SSD Inception V2                 | 66.10 | 13474849   | 7594247747    | 284.512918   | 23.74428378       | 9.393405405             | 14.35087838     |


## Run
Just running:
```
 ts_real_time.py
```
