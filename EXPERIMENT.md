## 复赛

### test
| Detector          | Regressor                                      | conf thres | iou thres | score thre | input size | concat | feature dim | result   | notes                                                                                                                                       | analysis                                                                                                                                                                                  |
|-------------------|------------------------------------------------|------------|-----------|------------|------------|--------|-------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| yolov5+multi_cls  | efficientnetb4_99.95_0.592392.ckpt             | 0.3        | 0.4       | 0.0        | 112        | True   | 256         | 0.810533 | 直接用初赛模型验证                                                                                                                          | 指标下降明显，可能是复赛数据集多了一些类导致度量效果差，也有可能是检测器效果也变差了                                                                                                      |
| yolov5+multi_cls  | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.3        | 0.4       | 0.3064     | 112        | True   | 256         | 0.829139(+1.86) | 考虑到可能是之前模型没有设置阈值, 由于之前训练模型的时候度量模型最佳阈值的选取有些问题，所以使用新模型＋最佳阈值测试                        | 新模型效果较好                                                                                                                                                                            |
| yolov5+multi_cls  | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.3        | 0.4       | 0.0        | 112        | True   | 256         | 0.829139(+0) | 考虑到可能最佳阈值去掉了一些预测正确的                                                                                                      | 说明模型预测score比较高，在0~0.3之间没有预测                                                                                                                                              |
| yolov5+multi_cls  | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.001      | 0.4       | 0.0        | 112        | True   | 256         | 0.836435(+0.73) | 考虑到对于新的数据，检测器有漏检, 将检测器阈值调低，提高召回率                                                                              | 指标上升，确实模型有漏检                                                                                                                                                                  |
| yolov5+multi_cls  | swintransformer+circleloss_99.9792_0.2760.ckpt | 0.001      | 0.4       | 0.2760        | 112        | True   | 256         | 0.754821(-8.16) | 尝试新的swin transformer模型                                                                                                                | 指标下降, 虽然swin transofmer在test上准确率较高，可能是transformer的建模能力对数据过拟合了，对于复赛数据集中潜在的未知数据识别能力差                                                                                                                                                                                  |
| yolov5+single_cls | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.001      | 0.4       | 0.0        | 112        | True   | 256         | 0.848588(+1.21) | 考虑到检测器漏检可能有一定程度是由于之前的模型为多类训练的                                                                                  | 指标上升，单类模型更好                                                                                                                                                                    |
| yolov5+single_cls | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.001      | 0.4       | 0.5        | 112        | True   | 256         | 0.842674(-0.59) | 验证是否度量不同商品的score值(相似度)很高，有误识别，所以提高score阈值为0.5                                                                 | 指标下降，说明这样反而去掉了一些正确的预测                                                                                                                                                |
| yolov5+single_cls | efficientnetb4+circleloss_99.9292_0.3064.ckpt  | 0.001      | 0.65      | 0.0        | 112        | True   | 256         | 0.848142(-0.04) | 为得到更多的预测框, 提高iou阈值，nms去掉更少的框                                                                                            | 指标下降，说明多的框反而是质量不好的                                                                                                                                                      |
| yolov5+single_cls | efficientb4_99.8333_0.1953.ckpt                | 0.001      | 0.4       | 0.0        | 112        | True   | 512         | 0.782949(-6.56) | 新模型，采用了更多的数据增强，特征维度增加到512, 去掉concat                                                                                 | 本意为去掉concat，inference时也错误的设置concat=True，但训练时错误的设置concat=True，所以影响了最佳模型的选取, 所以指标下降, 但是感觉应该每不会下降这么多                                                               |
| yolov5+single_cls | epoch-019_99.95_0.2291.ckpt                    | 0.001      | 0.4       | 0.0        | 112        | True   | 512         | 0.856337(+0.77) | 新模型，采用了更多的数据增强，特征维度增加到512, 去掉concat                                                                                 | 本意为去掉concat，inference时也错误的设置concat=True，训练时设置concat=False正确，所以最佳模型选取ok, 指标上升                                                                            |
| yolov5+single_cls | epoch-126_99.91_0.1663.ckpt                    | 0.001      | 0.4       | 0.0        | 112        | True   | 512         | 0.845064(-1.12) | 该模型为与上一条作对比，因为从测试精度来看，新模型收敛较快，之前的模型都在几十轮甚至100轮以上达到最佳精度, 验证新模型的test精度是否有代表性 | 本意为去掉concat，inference时也错误的设置concat=True，训练时设置concat=False正确，指标下降, 说明test精度还是有代表性                                                                      |
| yolov5+single_cls | epoch-019_99.95_0.2291.ckpt                    | 0.001      | 0.4       | 0.0        | 112        | False  | 512         | 0.854652(-0.168) | 由于上面inference时错误把concat=True，这里为验证112的新模型                                                                                 | 指标下降, 比concat=True时指标差一些，说明在正确选取模型之后(训练时concat=False)，inference时concat=True可能会提高精度                                     |
| yolov5+single_cls | epoch006_99.94_0.2488_224×224.ckpt             | 0.001      | 0.4       | 0.0        | 112        | True   | 512         | 0.733208(-12.3) | input-size设置为224, 训练时concat=False, 验证提高input-size的影响                                                                           | 本意为去掉concat，inference时也错误的设置concat=True，训练时设置concat=False正确，加上上面验证test精度有一定代表性，所以最佳模型选取ok, 指标下降，是由于inference时错误的把input-size=112 |
| yolov5+single_cls | epoch006_99.94_0.2488_224×224.ckpt             | 0.001      | 0.4       | 0.0        | 224        | False  | 512         | 0.873728(+1.74) | input-size设置为224, 训练时concat=False, 验证提高input-size的影响                                                                           | inference时concat=True，input-size=224, 指标上升明显，比input-size=112+concat推理高1.74，说明增大输入分辨率有一定效果                                                                                                        |

**Tips:每次对比是取的上一次的最好模型作为baseline**


| Detector          | Regressor                                                  | conf thres | iou thres | score thre | input size | concat | feature dim | result           | notes                                                                                 | analysis                                                    |
|-------------------|------------------------------------------------------------|------------|-----------|------------|------------|--------|-------------|------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------|
| yolov5+single_cls | 300epoch_swin_cirleloss99.90_0.2846.ckpt                   | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.893133(+1.94)  | 训练swin+224, 调整训练策略，降低学习率，使用余弦退火                                  | 训练时和inference时concat都为True, swin模型确实效果更好一些 |
| yolov5+single_cls | 300epoch_swin_cirleloss99.90_0.2846.ckpt                   | 0.001      | 0.4       | 0.2846     | 224        | True   | 512         | 0.893133(+0)     | 验证阈值                                                                              | 结果还是和之前一样，不变                                    |
| yolov5+single_cls | 300epoch_swin_cirleloss99.90_0.2846.ckpt                   | 0.001      | 0.4       | 0.0        | 224        | False  | 512         | 0.891848(-0.128) | 验证concat对swin的影响                                                                | 结果和之前eff+224一样，concat有提升                         |
| yolov5+single_cls | 019eopch_efficientb4_circleloss_99.947_0.3195_384×384.ckpt | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.866985(-2.61)  | input-size=384，efficientnet-b4 + concat效果并不好, 训练时和inference时concat都为True | 可能是concat对eff不友好(之前结果也表示)，或者是384效果不好  |



| Detector    | Regressor                                | conf thres | iou thres | score thre | input size | concat | feature dim | result            | notes                               | analysis                                                                                                        |
|-------------|------------------------------------------|------------|-----------|------------|------------|--------|-------------|-------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| yolov5x_PRC | 300epoch_swin_cirleloss99.90_0.2846.ckpt | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.893283(+0.143)  | 随机选取3000RPC的数据加入训练检测器 | 指标有一点上升，表示有正向作用，感觉提升不大，可能检测器已经差不多瓶颈了                                        |
| yolov5x_PRC | swin_large_028epoch_99.97_0.3506.ckpt    | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.916003(+2.272)  | 训练swin-large, 224, concat=True    | 指标明显提升，较大的模型确实有比较好的效果，也表明精度很大可能差在度量器上                                      |
| yolov5x_PRC | swin_large_028epoch_99.97_0.3506.ckpt    | 0.001      | 0.4       | 0.3506     | 224        | True   | 512         | 0.916003(+0.0)    | 对比阈值                            | 还是和之前的实验一样，得分不变                                                                                  |
| yolov5x_PRC | swin_large_028epoch_99.97_0.3506.ckpt    | 0.001      | 0.4       | 0.4506     | 224        | True   | 512         | 0.915952(-0.0051) | 对比阈值                            | 得分少量减少，说明测试集大概率没有其他类别，因为0和0.35的指标一样，而再提高阈值少量降低，说明提高阈值导致漏检了 |  |


| Detector    | Regressor                             | conf thres | iou thres | score thre | input size | concat | feature dim | result   | notes                                                  | analysis                    |
|-------------|---------------------------------------|------------|-----------|------------|------------|--------|-------------|----------|--------------------------------------------------------|-----------------------------|
| yolov5x_PRC | swin_large_028epoch_99.97_0.3506.ckpt | 0.1        | 0.4       | 0.0        | 224        | True   | 512         | 0.915654 | 对比检测器阈值                                         | 结果显示阈值设置为0.001最佳 |
| yolov5x_PRC | swin_large_epoch93_99.9733.ckpt       | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.914813 | 对比在测试集上精度更高的模型                           | 指标略微下降                |
| yolov5x_PRC | swin_small_cgd_epoch118_99.92.ckpt    | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.894241 | 更换了训练策略，学习率的small模型，对比之前的small模型, 本意为使用cgd，但是未开启 | 指标略微上升                |


| Detector    | Regressor                              | conf thres | iou thres | score thre | input size | concat | feature dim | result   | notes                                                                      | analysis                                                                            |
|-------------|----------------------------------------|------------|-----------|------------|------------|--------|-------------|----------|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| yolov5x_PRC | swin_large_cdg_epoch034_99.9967.ckpt   | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.915495 | swin large+cgd                                                             | 指标下降                                                                            |
| yolov5x_PRC | swin_small_cgd_epoch040._9.9633.ckpt   | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.89661  | swin small+cgd                                                             | 比不加cgd的small模型指标有上升                                                      |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt     | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.916085 | swin large+cgd更换一个epoch的模型                                          | 指标微微提升                                                                        |
| yolov5x_PRC | swin_large_384_cgd_epoch038_99.92.ckpt | 0.001      | 0.4       | 0.0        | 384        | False  | 512         | 0.818517 | swin large+cgd + 384, 由于384会超时，所以去掉concat再增加batchsize才出结果 | 指标明显下降，可以放弃384了，之前eff+384也有精度下降                                |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt     | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.915879 | 对图片进行上下翻转，左右翻转，上下左右翻转再拼接为2048维度的预测           | 指标微微下降，说明一味的 进行翻转再concat也不一定会提高指标，之后可以试一下使用均值 |



| Detector    | Regressor                          | conf thres | iou thres | score thre | input size | concat | feature dim | result   | notes                                                                                | analysis     |
|-------------|------------------------------------|------------|-----------|------------|------------|--------|-------------|----------|--------------------------------------------------------------------------------------|--------------|
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt | 0.001      | 0.4+wbf   | 0.0        | 224        | True   | 512         | 0.911388 | 尝试在检测器中使用WBF                                                                | 指标下降     |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt | 0.001      | 0.4       | 0.0        | 224        | mean   | 512         | 0.915829 | 特征融合方式不使用concat而是求均值                                                   | 指标微微下降 |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt | 0.001      | 0.5       | 0.0        | 224        | True   | 512         | 0.916196 | 尝试不同的iou阈值                                                                    | 指标微微上升 |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt | 0.001      | 0.5+0.85  | 0.0        | 224        | True   | 512         | 0.915685 | 发现预测框中有重复的nms为去掉的框，所以添加了一个阈值为0.85的nms(交集/较小的box面积) | 指标微微下降 |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt | 0.01       | 0.5       | 0.0        | 224        | True   | 512         | 0.916008 | 尝试不同的conf阈值                                                                   | 指标微微下降 |
**notes** :目前最好的配置是0.001+0.5+0.0+concat;


| Detector    | Regressor                          | conf thres | iou thres | score thre | input size | concat | feature dim | result   | notes                                        | analysis                                        |
|-------------|------------------------------------|------------|-----------|------------|------------|--------|-------------|----------|----------------------------------------------|-------------------------------------------------|
| yolov5x_PRC | swin_large_cdg_epoch055_91.55.ckpt | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.909909 | 清理数据集，将一些重复的或者是类似的数据删除 | 指标下降,  可能是由于删除了数据集，所以指标下降 |
| yolov5x_PRC | 078.ckpt                           | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.907079 | epoch78的模型，对比上一条                    | 指标下降,  指标下降                             |



| Detector    | Regressor                                | conf thres | iou thres | score thre | input size | concat | feature dim | result   | notes                                                                          | analysis                                                                                       |
|-------------|------------------------------------------|------------|-----------|------------|------------|--------|-------------|----------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| yolov5x_PRC | swin_large_cgd_epoch040_311.ckpt         | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.901047 | 清理数据集，将类似的数据融合而不是删除，且清洗一些噪声数据，再加入了5w旷世数据 | 指标下降                                                                                       |
| yolov5x_PRC | swin_large_cdg_epoch053_0.98745_311.ckpt | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.90879  | 对比上一条，epoch为53                                                          | 指标相比于上一条上升                                                                           |
| yolov5x_PRC | swin_large_cgd_epoch061_0.98625_311.ckpt | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.910707 | 对比上一条，epoch为61                                                          | 指标相比于上一条上升，在测试集上上一条更高，但是损失这一条更低，推测可能通过loss选取模型更靠谱 |
| yolov5x_PRC | 085_311.ckpt                             | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.914581 | 对比上一条，epoch为85                                                          | 指标相比于上一条上升，在测试集上上一条更高，但是损失这一条更低，推测可能通过loss选取模型更靠谱 |
| yolov5x_PRC | 087_311.ckpt                             | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.914483 | 对比上一条，epoch为87                                                          | 指标相比于上一条下降                                                                           |
| yolov5x_PRC | swin_large_cdg_epoch122_98.92_311.ckpt   | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.915024 | 对比上一条，epoch为122                                                         | 指标相比于上一条上升，在测试集上上一条更高，但是损失这一条更低，推测可能通过loss选取模型更靠谱 |
| yolov5x_PRC | swin_large_epoch133_98.75_311.ckpt       | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.915465 | 对比上一条，epoch为133                                                         | 指标相比于上一条上升，在测试集上上一条更高，但是损失这一条更低，推测可能通过loss选取模型更靠谱 |
| yolov5x_PRC | 150_311.ckpt                             | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.915279 | 对比上一条，epoch为150                                                         | 指标相比于上一条下降                                                                           |

| Detector    | Regressor         | conf thres | iou thres | score thre | input size | concat | feature dim | result   | notes                                                 | analysis                                   |
|-------------|-------------------|------------|-----------|------------|------------|--------|-------------|----------|-------------------------------------------------------|--------------------------------------------|
| yolov5x_PRC | 069_111_1024.ckpt | 0.001      | 0.4       | 0.0        | 224        | True   | 1024        | 0.915097 | 尝试增加feature维度，与之前模型不同，这个模型收敛很慢 | 指标下降，等再训练一段时间收敛了在测试一下 |


## 模型融合
| Detector    | Regressor                                                                                                     | conf thres | iou thres | score thre | input size | concat | feature dim | result   | notes                                                                                                                                                  | analysis                                       |
|-------------|---------------------------------------------------------------------------------------------------------------|------------|-----------|------------|------------|--------|-------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt+swin_large_028epoch_99.97_0.3506.ckpt                                      | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.916344 | 尝试模型融合，目前精度最好的两个模型的预测features取均值                                                                                               | 指标稍稍上升                                   |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt+swin_large_028epoch_99.97_0.3506.ckpt                                      | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.916676 | 尝试模型融合，目前精度最好的两个模型的预测中取相似度更高的一个作为预测                                                                                 | 指标稍稍上升                                   |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt+swin_large_028epoch_99.97_0.3506.ckpt+swin_small_cgd_epoch040._9.9633.ckpt | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.919159 | 尝试模型融合，目前精度最好的两个large模型和一个small模型, 进行投票，如果票数一样则选取相似度最高的，否则选择票数最多的                                 | 指标稍稍上升，目前最高                         |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt+swin_large_028epoch_99.97_0.3506.ckpt+swin_small_cgd_epoch040._9.9633.ckpt | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.918895 | 尝试模型融合，目前精度最好的两个large模型和一个small模型, 进行投票，如果票数一样则选取精度最高模型的预测，否则选择票数最多的                           | 该策略比上一条策略低一些                       |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt+epoch006_99.94_0.2488_224×224.ckpt+swin_small_cgd_epoch040._9.9633.ckpt    | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.915433 | 尝试模型融合，目前精度最好的一个large模型和一个small模型,eff模型 进行投票，如果票数一样则选取相似度最高的，否则选择票数最多的                          | 该策略比之前的策略低,应该是eff效果模型效果较差 |
| yolov5x_PRC | siwn_large_cgd_epoch049_99.99.ckpt+swin_large_epoch133_98.75_311.ckpt+swin_small_cgd_epoch040._9.9633.ckpt    | 0.001      | 0.4       | 0.0        | 224        | True   | 512         | 0.917459 | 尝试模型融合，目前精度最好的一个large模型和一个small模型,和增加了旷世数据集训练的large模型进行投票，如果票数一样则选取相似度最高的，否则选择票数最多的 | 该策略比之前的策略低,应该是eff效果模型效果较差 |





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


