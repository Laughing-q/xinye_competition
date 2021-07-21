## 复赛

### test
| Detector          | Regressor                                      | conf thres | iou thres | score thre | input size | concat | feature dim | result   | notes                                                                                                                                       | analysis                                                                                                                                                                                  |
|-------------------|------------------------------------------------|------------|-----------|------------|------------|--------|-------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| yolov5+multi_cls  | efficientnetb4_99.95_0.592392.ckpt             | 0.3        | 0.4       | 0.0        | 112        | True   | 256         | 0.810533 | 直接用初赛模型验证                                                                                                                          | 指标下降明显，可能是复赛数据集多了一些类导致度量效果差，也有可能是检测器效果也变差了                                                                                                      |
| yolov5+multi_cls  | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.3        | 0.4       | 0.3064     | 112        | True   | 256         | 0.829139(+1.86) | 考虑到可能是之前模型没有设置阈值, 由于之前训练模型的时候度量模型最佳阈值的选取有些问题，所以使用新模型＋最佳阈值测试                        | 新模型效果较好                                                                                                                                                                            |
| yolov5+multi_cls  | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.3        | 0.4       | 0.0        | 112        | True   | 256         | 0.829139(+0) | 考虑到可能最佳阈值去掉了一些预测正确的                                                                                                      | 说明模型预测score比较高，在0~0.3之间没有预测                                                                                                                                              |
| yolov5+multi_cls  | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.001      | 0.4       | 0.0        | 112        | True   | 256         | 0.836435(+0.73) | 考虑到对于新的数据，检测器有漏检, 将检测器阈值调低，提高召回率                                                                              | 指标上升，确实模型有漏检                                                                                                                                                                  |
| yolov5+multi_cls  | swintransformer+circleloss_99.9792_0.2760.ckpt | 0.001      | 0.4       | 0.0        | 112        | True   | 256         | 0.754821(-8.16) | 尝试新的swin transformer模型                                                                                                                | 指标下降, 虽然swin transofmer在test上准确率较高，可能是transformer的建模能力对数据过拟合了，对于复赛数据集中潜在的未知数据识别能力差                                                                                                                                                                                  |
| yolov5+single_cls | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.001      | 0.4       | 0.0        | 112        | True   | 256         | 0.848588(+1.21) | 考虑到检测器漏检可能有一定程度是由于之前的模型为多类训练的                                                                                  | 指标上升，单类模型更好                                                                                                                                                                    |
| yolov5+single_cls | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.001      | 0.4       | 0.5        | 112        | True   | 256         | 0.842674(-0.59) | 验证是否度量不同商品的score值(相似度)很高，有误识别，所以提高score阈值为0.5                                                                 | 指标下降，说明这样反而去掉了一些正确的预测                                                                                                                                                |
| yolov5+single_cls | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.001      | 0.65      | 0.0        | 112        | True   | 256         | 0.848142(-0.04) | 为得到更多的预测框, 提高iou阈值，nms去掉更少的框                                                                                            | 指标下降，说明多的框反而是质量不好的                                                                                                                                                      |
| yolov5+single_cls | efficientb4_99.8333_0.1953.ckpt                | 0.001      | 0.4       | 0.0        | 112        | True   | 512         | 0.782949(-6.56) | 新模型，采用了更多的数据增强，特征维度增加到512, 去掉concat                                                                                 | 本意为去掉concat，inference时也错误的设置concat=True，但训练时错误的设置concat=True，所以影响了最佳模型的选取, 所以指标下降                                                               |
| yolov5+single_cls | epoch-019_99.95_0.2291.ckpt                    | 0.001      | 0.4       | 0.0        | 112        | True   | 512         | 0.856337(+0.77) | 新模型，采用了更多的数据增强，特征维度增加到512, 去掉concat                                                                                 | 本意为去掉concat，inference时也错误的设置concat=True，训练时设置concat=False正确，所以最佳模型选取ok, 指标上升                                                                            |
| yolov5+single_cls | epoch-126_99.91_0.1663.ckpt                    | 0.001      | 0.4       | 0.0        | 112        | True   | 512         | 0.845064(-1.12) | 该模型为与上一条作对比，因为从测试精度来看，新模型收敛较快，之前的模型都在几十轮甚至100轮以上达到最佳精度, 验证新模型的test精度是否有代表性 | 本意为去掉concat，inference时也错误的设置concat=True，训练时设置concat=False正确，指标下降, 说明test精度还是有代表性                                                                      |
| yolov5+single_cls | epoch006_99.94_0.2488_224×224.ckpt             | 0.001      | 0.4       | 0.0        | 112        | True   | 512         | 0.733208(-12.3) | input-size设置为224, 训练时concat=False, 验证提高input-size的影响                                                                           | 本意为去掉concat，inference时也错误的设置concat=True，训练时设置concat=False正确，加上上面验证test精度有一定代表性，所以最佳模型选取ok, 指标下降，是由于inference时错误的把input-size=112 |
| yolov5+single_cls | epoch006_99.94_0.2488_224×224.ckpt             | 0.001      | 0.4       | 0.0        | 224        | False  | 512         | 0.873728(+1.74) | input-size设置为224, 训练时concat=False, 验证提高input-size的影响                                                                           | inference时concat=True，input-size=224, 指标上升明显，说明增大输入分辨率有一定效果                                                                                                        |
| yolov5+single_cls | epoch-019_99.95_0.2291.ckpt                    | 0.001      | 0.4       | 0.0        | 112        | False  | 512         | 0.854652(-0.168) | 由于上面inference时错误把concat=True，这里为验证112的新模型                                                                                 | 指标下降，说明增大输入分辨率有一定效果, 且比concat=True时指标差一些，说明在正确选取模型之后(训练时concat=False)，inference时concat=True可能会提高精度                                     |

**Tips:每次对比是取的上一次的最好模型作为baseline**

## 初赛
yolov5s+model1, detect_conf=0.3, iou=0.4, regress_score=0.43: 0.58514;

yolov5s+model1, detect_conf=0.3, iou=0.4, regress_score=0: 0.588509;

yolov5s+model1, detect_conf=0.1, iou=0.6, regress_score=0:0.588442 ;


yolov5s+model2, detect_conf=0.3, iou=0.4, regress_score=0.62: 0.879246

yolov5s+model2, detect_conf=0.3, iou=0.4, regress_score=0: 0.881367

yolov5l+model2, detect_conf=0.3, iou=0.4, regress_score=0:0.889525 

yolov5x+model2, detect_conf=0.3, iou=0.4, regress_score=0:0.896242

yolov5x+model2, detect_conf=0.001, iou=0.6, regress_score=0:0.896091


通过可视化预测发现，由于yolov5是按照多类别训练的，所以会有nms没去掉的重合度很高的框，所以这里将nms设置为agnosic；

yolov5x+model2, detect_conf=0.3, iou=0.4,agnosic, regress_score=0:0.898882

yolov5x+model2, detect_conf=0.3, iou=0.6,agnosic, regress_score=0:0.898572

yolov5x+resnet50, detect_conf=0.3, iou=0.4,agnosic, regress_score=0:0.907966

a_prediction_regressthres_conf_iou

a_prediction_0_0.3_0.4_x_agnostic_resnet152d.json: 0.906644

a_prediction_0.44672_0.3_0.4_x_agnostic_resnet152d.json: 0.906116

a_prediction_0_0.3_0.4_x_agnostic_efficientnet3.json:0.930652

a_prediction_0.592392_0.3_0.4_x_agnostic_efficientnet3.json: 0.927049

a_prediction_0_0.3_0.4_x_agnostic_efficientnet2.json: 0.92878

a_prediction_0.475603_0.3_0.4_x_agnostic_efficientnet2.json：0.927674


**复审**：0.931225


