
# CIFAR-100 State-of-Art Model
----

In this example, we will show a new state-of-art result on CIFAR-100.
We use a sub-Inception Network with Randomized ReLU (RReLU), and achieved 75.68% accuracy on CIFAR-100.

We trained from raw pixel directly, only random crop from 3x28x28 from original 3x32x32 image with random flip, which is same to other experiments. 

We don't do any parameter search, all hyper-parameters come from ImageNet experience, and this work is just for fun. Definitely you can improve it.

Train this network requires 3796MB GPU Memory.

----


| Model                       | Test Accuracy |
| --------------------------- | ------------- |
| **Sub-Inception + RReLU** [1], [2]   | **75.68%**       |
| Highway Network  [3] | 67.76%        |
| Deeply Supervised Network [4]   | 65.43%        |




```python
import mxnet as mx
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
```

Next step we will set up basic Factories for Inception


```python
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.LeakyReLU(data=bn, act_type='rrelu', name='rrelu_%s%s' %(name, suffix))
    return act

def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def InceptionFactoryB(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name):
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1),  name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type="max", name=('max_pool_%s_pool' % name))
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat
```

Build Network by using Factories


```python
def inception(nhidden, grad_scale):
    # data
    data = mx.symbol.Variable(name="data")
    # stage 2
    in3a = InceptionFactoryA(data, 64, 64, 64, 64, 96, "avg", 32, '3a')
    in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
    in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, '3c')
    # stage 3
    in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
    in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
    in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
    in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
    in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, '4e')
    # stage 4
    in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
    in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
    # global avg pooling
    avg = mx.symbol.Pooling(data=in5b, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')
    # linear classifier
    flatten = mx.symbol.Flatten(data=avg, name='flatten')
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=nhidden, name='fc')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax

softmax = inception(100, 1.0)
```

Make data iterator. Note we convert original CIFAR-100 dataset into image format then pack into RecordIO in purpose of using our build-in image augmentation. For details about RecordIO, please refer ()[]


```python
batch_size = 64

train_dataiter = mx.io.ImageRecordIter(
    shuffle=True,
    path_imgrec="./data/train.rec",
    mean_img="./data/mean.bin",
    rand_crop=True,
    rand_mirror=True,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2)

test_dataiter = mx.io.ImageRecordIter(
    path_imgrec="./data/test.rec",
    mean_img="./data/mean.bin",
    rand_crop=False,
    rand_mirror=False,
    data_shape=(3, 28, 28),
    batch_size=batch_size,
    prefetch_buffer=4,
    preprocess_threads=2,
    round_batch=False)

```

Make model


```python
num_epoch = 38
model_prefix = "model/cifar_100"

softmax = inception(100, 1.0)

model = mx.model.FeedForward(ctx=mx.gpu(), symbol=softmax, num_epoch=num_epoch,
                             learning_rate=0.05, momentum=0.9, wd=0.0001)

```

Fit first stage


```python
model.fit(X=train_dataiter,
          eval_data=test_dataiter,
          eval_metric="accuracy",
          batch_end_callback=mx.callback.Speedometer(batch_size, 200),
          epoch_end_callback=mx.callback.do_checkpoint(model_prefix))

```

    INFO:root:Start training with [gpu(0)]
    INFO:root:Batch [200]	Speed: 157.49 samples/sec
    INFO:root:Batch [400]	Speed: 144.49 samples/sec
    INFO:root:Batch [600]	Speed: 142.79 samples/sec
    INFO:root:Iteration[0] Train-accuracy=0.184423
    INFO:root:Iteration[0] Time cost=342.269
    INFO:root:Iteration[0] Validation-accuracy=0.279757
    INFO:root:Saved checkpoint to "model/cifar_100-0001.params"
    INFO:root:Batch [200]	Speed: 142.65 samples/sec
    INFO:root:Batch [400]	Speed: 141.95 samples/sec
    INFO:root:Batch [600]	Speed: 141.44 samples/sec
    INFO:root:Iteration[1] Train-accuracy=0.363516
    INFO:root:Iteration[1] Time cost=352.763
    INFO:root:Iteration[1] Validation-accuracy=0.421775
    INFO:root:Saved checkpoint to "model/cifar_100-0002.params"
    INFO:root:Batch [200]	Speed: 142.58 samples/sec
    INFO:root:Batch [400]	Speed: 141.47 samples/sec
    INFO:root:Batch [600]	Speed: 141.47 samples/sec
    INFO:root:Iteration[2] Train-accuracy=0.464609
    INFO:root:Iteration[2] Time cost=353.075
    INFO:root:Iteration[2] Validation-accuracy=0.501891
    INFO:root:Saved checkpoint to "model/cifar_100-0003.params"
    INFO:root:Batch [200]	Speed: 142.24 samples/sec
    INFO:root:Batch [400]	Speed: 141.21 samples/sec
    INFO:root:Batch [600]	Speed: 141.11 samples/sec
    INFO:root:Iteration[3] Train-accuracy=0.529690
    INFO:root:Iteration[3] Time cost=353.893
    INFO:root:Iteration[3] Validation-accuracy=0.548368
    INFO:root:Saved checkpoint to "model/cifar_100-0004.params"
    INFO:root:Batch [200]	Speed: 142.36 samples/sec
    INFO:root:Batch [400]	Speed: 141.10 samples/sec
    INFO:root:Batch [600]	Speed: 141.13 samples/sec
    INFO:root:Iteration[4] Train-accuracy=0.572331
    INFO:root:Iteration[4] Time cost=354.524
    INFO:root:Iteration[4] Validation-accuracy=0.588973
    INFO:root:Saved checkpoint to "model/cifar_100-0005.params"
    INFO:root:Batch [200]	Speed: 141.93 samples/sec
    INFO:root:Batch [400]	Speed: 141.05 samples/sec
    INFO:root:Batch [600]	Speed: 141.14 samples/sec
    INFO:root:Iteration[5] Train-accuracy=0.610455
    INFO:root:Iteration[5] Time cost=354.183
    INFO:root:Iteration[5] Validation-accuracy=0.604299
    INFO:root:Saved checkpoint to "model/cifar_100-0006.params"
    INFO:root:Batch [200]	Speed: 141.44 samples/sec
    INFO:root:Batch [400]	Speed: 140.94 samples/sec
    INFO:root:Batch [600]	Speed: 140.65 samples/sec
    INFO:root:Iteration[6] Train-accuracy=0.634563
    INFO:root:Iteration[6] Time cost=355.035
    INFO:root:Iteration[6] Validation-accuracy=0.598229
    INFO:root:Saved checkpoint to "model/cifar_100-0007.params"
    INFO:root:Batch [200]	Speed: 141.57 samples/sec
    INFO:root:Batch [400]	Speed: 140.94 samples/sec
    INFO:root:Batch [600]	Speed: 140.60 samples/sec
    INFO:root:Iteration[7] Train-accuracy=0.662832
    INFO:root:Iteration[7] Time cost=355.113
    INFO:root:Iteration[7] Validation-accuracy=0.618432
    INFO:root:Saved checkpoint to "model/cifar_100-0008.params"
    INFO:root:Batch [200]	Speed: 141.51 samples/sec
    INFO:root:Batch [400]	Speed: 140.69 samples/sec
    INFO:root:Batch [600]	Speed: 140.63 samples/sec
    INFO:root:Iteration[8] Train-accuracy=0.677390
    INFO:root:Iteration[8] Time cost=355.880
    INFO:root:Iteration[8] Validation-accuracy=0.631270
    INFO:root:Saved checkpoint to "model/cifar_100-0009.params"
    INFO:root:Batch [200]	Speed: 141.22 samples/sec
    INFO:root:Batch [400]	Speed: 140.60 samples/sec
    INFO:root:Batch [600]	Speed: 140.58 samples/sec
    INFO:root:Iteration[9] Train-accuracy=0.695923
    INFO:root:Iteration[9] Time cost=355.619
    INFO:root:Iteration[9] Validation-accuracy=0.639431
    INFO:root:Saved checkpoint to "model/cifar_100-0010.params"
    INFO:root:Batch [200]	Speed: 141.21 samples/sec
    INFO:root:Batch [400]	Speed: 140.70 samples/sec
    INFO:root:Batch [600]	Speed: 140.57 samples/sec
    INFO:root:Iteration[10] Train-accuracy=0.712428
    INFO:root:Iteration[10] Time cost=355.604
    INFO:root:Iteration[10] Validation-accuracy=0.651373
    INFO:root:Saved checkpoint to "model/cifar_100-0011.params"
    INFO:root:Batch [200]	Speed: 141.07 samples/sec
    INFO:root:Batch [400]	Speed: 140.58 samples/sec
    INFO:root:Batch [600]	Speed: 140.48 samples/sec
    INFO:root:Iteration[11] Train-accuracy=0.729473
    INFO:root:Iteration[11] Time cost=355.796
    INFO:root:Iteration[11] Validation-accuracy=0.645502
    INFO:root:Saved checkpoint to "model/cifar_100-0012.params"
    INFO:root:Batch [200]	Speed: 141.22 samples/sec
    INFO:root:Batch [400]	Speed: 140.43 samples/sec
    INFO:root:Batch [600]	Speed: 140.54 samples/sec
    INFO:root:Iteration[12] Train-accuracy=0.739051
    INFO:root:Iteration[12] Time cost=356.473
    INFO:root:Iteration[12] Validation-accuracy=0.663217
    INFO:root:Saved checkpoint to "model/cifar_100-0013.params"
    INFO:root:Batch [200]	Speed: 141.15 samples/sec
    INFO:root:Batch [400]	Speed: 140.58 samples/sec
    INFO:root:Batch [600]	Speed: 140.45 samples/sec
    INFO:root:Iteration[13] Train-accuracy=0.752821
    INFO:root:Iteration[13] Time cost=355.815
    INFO:root:Iteration[13] Validation-accuracy=0.653961
    INFO:root:Saved checkpoint to "model/cifar_100-0014.params"
    INFO:root:Batch [200]	Speed: 140.89 samples/sec
    INFO:root:Batch [400]	Speed: 140.35 samples/sec
    INFO:root:Batch [600]	Speed: 140.45 samples/sec
    INFO:root:Iteration[14] Train-accuracy=0.759083
    INFO:root:Iteration[14] Time cost=356.155
    INFO:root:Iteration[14] Validation-accuracy=0.661027
    INFO:root:Saved checkpoint to "model/cifar_100-0015.params"
    INFO:root:Batch [200]	Speed: 141.13 samples/sec
    INFO:root:Batch [400]	Speed: 140.52 samples/sec
    INFO:root:Batch [600]	Speed: 140.38 samples/sec
    INFO:root:Iteration[15] Train-accuracy=0.770367
    INFO:root:Iteration[15] Time cost=355.945
    INFO:root:Iteration[15] Validation-accuracy=0.669984
    INFO:root:Saved checkpoint to "model/cifar_100-0016.params"
    INFO:root:Batch [200]	Speed: 141.21 samples/sec
    INFO:root:Batch [400]	Speed: 140.57 samples/sec
    INFO:root:Batch [600]	Speed: 140.44 samples/sec
    INFO:root:Iteration[16] Train-accuracy=0.781030
    INFO:root:Iteration[16] Time cost=356.440
    INFO:root:Iteration[16] Validation-accuracy=0.661027
    INFO:root:Saved checkpoint to "model/cifar_100-0017.params"
    INFO:root:Batch [200]	Speed: 141.00 samples/sec
    INFO:root:Batch [400]	Speed: 140.44 samples/sec
    INFO:root:Batch [600]	Speed: 140.44 samples/sec
    INFO:root:Iteration[17] Train-accuracy=0.787232
    INFO:root:Iteration[17] Time cost=356.027
    INFO:root:Iteration[17] Validation-accuracy=0.676652
    INFO:root:Saved checkpoint to "model/cifar_100-0018.params"
    INFO:root:Batch [200]	Speed: 140.77 samples/sec
    INFO:root:Batch [400]	Speed: 140.56 samples/sec
    INFO:root:Batch [600]	Speed: 140.45 samples/sec
    INFO:root:Iteration[18] Train-accuracy=0.796975
    INFO:root:Iteration[18] Time cost=356.066
    INFO:root:Iteration[18] Validation-accuracy=0.678145
    INFO:root:Saved checkpoint to "model/cifar_100-0019.params"
    INFO:root:Batch [200]	Speed: 141.01 samples/sec
    INFO:root:Batch [400]	Speed: 140.42 samples/sec
    INFO:root:Batch [600]	Speed: 140.42 samples/sec
    INFO:root:Iteration[19] Train-accuracy=0.805378
    INFO:root:Iteration[19] Time cost=356.019
    INFO:root:Iteration[19] Validation-accuracy=0.677548
    INFO:root:Saved checkpoint to "model/cifar_100-0020.params"
    INFO:root:Batch [200]	Speed: 141.17 samples/sec
    INFO:root:Batch [400]	Speed: 140.52 samples/sec
    INFO:root:Batch [600]	Speed: 140.57 samples/sec
    INFO:root:Iteration[20] Train-accuracy=0.808903
    INFO:root:Iteration[20] Time cost=356.454
    INFO:root:Iteration[20] Validation-accuracy=0.665207
    INFO:root:Saved checkpoint to "model/cifar_100-0021.params"
    INFO:root:Batch [200]	Speed: 140.94 samples/sec
    INFO:root:Batch [400]	Speed: 140.29 samples/sec
    INFO:root:Batch [600]	Speed: 140.28 samples/sec
    INFO:root:Iteration[21] Train-accuracy=0.815761
    INFO:root:Iteration[21] Time cost=356.311
    INFO:root:Iteration[21] Validation-accuracy=0.671377
    INFO:root:Saved checkpoint to "model/cifar_100-0022.params"
    INFO:root:Batch [200]	Speed: 140.83 samples/sec
    INFO:root:Batch [400]	Speed: 140.27 samples/sec
    INFO:root:Batch [600]	Speed: 140.30 samples/sec
    INFO:root:Iteration[22] Train-accuracy=0.822803
    INFO:root:Iteration[22] Time cost=356.518
    INFO:root:Iteration[22] Validation-accuracy=0.670482
    INFO:root:Saved checkpoint to "model/cifar_100-0023.params"
    INFO:root:Batch [200]	Speed: 141.10 samples/sec
    INFO:root:Batch [400]	Speed: 140.45 samples/sec
    INFO:root:Batch [600]	Speed: 140.24 samples/sec
    INFO:root:Iteration[23] Train-accuracy=0.827545
    INFO:root:Iteration[23] Time cost=356.127
    INFO:root:Iteration[23] Validation-accuracy=0.671178
    INFO:root:Saved checkpoint to "model/cifar_100-0024.params"
    INFO:root:Batch [200]	Speed: 140.91 samples/sec
    INFO:root:Batch [400]	Speed: 140.28 samples/sec
    INFO:root:Batch [600]	Speed: 140.30 samples/sec
    INFO:root:Iteration[24] Train-accuracy=0.833760
    INFO:root:Iteration[24] Time cost=356.974
    INFO:root:Iteration[24] Validation-accuracy=0.670382
    INFO:root:Saved checkpoint to "model/cifar_100-0025.params"
    INFO:root:Batch [200]	Speed: 141.00 samples/sec
    INFO:root:Batch [400]	Speed: 140.41 samples/sec
    INFO:root:Batch [600]	Speed: 140.41 samples/sec
    INFO:root:Iteration[25] Train-accuracy=0.839609
    INFO:root:Iteration[25] Time cost=356.055
    INFO:root:Iteration[25] Validation-accuracy=0.677946
    INFO:root:Saved checkpoint to "model/cifar_100-0026.params"
    INFO:root:Batch [200]	Speed: 141.04 samples/sec
    INFO:root:Batch [400]	Speed: 140.42 samples/sec
    INFO:root:Batch [600]	Speed: 140.43 samples/sec
    INFO:root:Iteration[26] Train-accuracy=0.841909
    INFO:root:Iteration[26] Time cost=356.007
    INFO:root:Iteration[26] Validation-accuracy=0.677946
    INFO:root:Saved checkpoint to "model/cifar_100-0027.params"
    INFO:root:Batch [200]	Speed: 140.88 samples/sec
    INFO:root:Batch [400]	Speed: 140.39 samples/sec
    INFO:root:Batch [600]	Speed: 140.14 samples/sec
    INFO:root:Iteration[27] Train-accuracy=0.846411
    INFO:root:Iteration[27] Time cost=356.341
    INFO:root:Iteration[27] Validation-accuracy=0.682126
    INFO:root:Saved checkpoint to "model/cifar_100-0028.params"
    INFO:root:Batch [200]	Speed: 141.16 samples/sec
    INFO:root:Batch [400]	Speed: 140.15 samples/sec
    INFO:root:Batch [600]	Speed: 139.99 samples/sec
    INFO:root:Iteration[28] Train-accuracy=0.847966
    INFO:root:Iteration[28] Time cost=357.334
    INFO:root:Iteration[28] Validation-accuracy=0.676652
    INFO:root:Saved checkpoint to "model/cifar_100-0029.params"
    INFO:root:Batch [200]	Speed: 140.74 samples/sec
    INFO:root:Batch [400]	Speed: 140.37 samples/sec
    INFO:root:Batch [600]	Speed: 140.35 samples/sec
    INFO:root:Iteration[29] Train-accuracy=0.860075
    INFO:root:Iteration[29] Time cost=356.321
    INFO:root:Iteration[29] Validation-accuracy=0.674363
    INFO:root:Saved checkpoint to "model/cifar_100-0030.params"
    INFO:root:Batch [200]	Speed: 140.75 samples/sec
    INFO:root:Batch [400]	Speed: 140.35 samples/sec
    INFO:root:Batch [600]	Speed: 140.34 samples/sec
    INFO:root:Iteration[30] Train-accuracy=0.856554
    INFO:root:Iteration[30] Time cost=356.349
    INFO:root:Iteration[30] Validation-accuracy=0.669686
    INFO:root:Saved checkpoint to "model/cifar_100-0031.params"
    INFO:root:Batch [200]	Speed: 141.06 samples/sec
    INFO:root:Batch [400]	Speed: 140.47 samples/sec
    INFO:root:Batch [600]	Speed: 140.46 samples/sec
    INFO:root:Iteration[31] Train-accuracy=0.861436
    INFO:root:Iteration[31] Time cost=355.920
    INFO:root:Iteration[31] Validation-accuracy=0.676254
    INFO:root:Saved checkpoint to "model/cifar_100-0032.params"
    INFO:root:Batch [200]	Speed: 141.03 samples/sec
    INFO:root:Batch [400]	Speed: 140.45 samples/sec
    INFO:root:Batch [600]	Speed: 140.28 samples/sec
    INFO:root:Iteration[32] Train-accuracy=0.858416
    INFO:root:Iteration[32] Time cost=357.042
    INFO:root:Iteration[32] Validation-accuracy=0.686405
    INFO:root:Saved checkpoint to "model/cifar_100-0033.params"
    INFO:root:Batch [200]	Speed: 140.66 samples/sec
    INFO:root:Batch [400]	Speed: 140.16 samples/sec
    INFO:root:Batch [600]	Speed: 140.12 samples/sec
    INFO:root:Iteration[33] Train-accuracy=0.868858
    INFO:root:Iteration[33] Time cost=356.653
    INFO:root:Iteration[33] Validation-accuracy=0.679041
    INFO:root:Saved checkpoint to "model/cifar_100-0034.params"
    INFO:root:Batch [200]	Speed: 140.62 samples/sec
    INFO:root:Batch [400]	Speed: 140.55 samples/sec
    INFO:root:Batch [600]	Speed: 140.50 samples/sec
    INFO:root:Iteration[34] Train-accuracy=0.870319
    INFO:root:Iteration[34] Time cost=356.121
    INFO:root:Iteration[34] Validation-accuracy=0.671676
    INFO:root:Saved checkpoint to "model/cifar_100-0035.params"
    INFO:root:Batch [200]	Speed: 140.81 samples/sec
    INFO:root:Batch [400]	Speed: 140.39 samples/sec
    INFO:root:Batch [600]	Speed: 140.39 samples/sec
    INFO:root:Iteration[35] Train-accuracy=0.874060
    INFO:root:Iteration[35] Time cost=356.216
    INFO:root:Iteration[35] Validation-accuracy=0.684813
    INFO:root:Saved checkpoint to "model/cifar_100-0036.params"
    INFO:root:Batch [200]	Speed: 141.11 samples/sec
    INFO:root:Batch [400]	Speed: 140.49 samples/sec
    INFO:root:Batch [600]	Speed: 140.48 samples/sec
    INFO:root:Iteration[36] Train-accuracy=0.872043
    INFO:root:Iteration[36] Time cost=356.771
    INFO:root:Iteration[36] Validation-accuracy=0.670581
    INFO:root:Saved checkpoint to "model/cifar_100-0037.params"
    INFO:root:Batch [200]	Speed: 140.93 samples/sec
    INFO:root:Batch [400]	Speed: 140.48 samples/sec
    INFO:root:Batch [600]	Speed: 140.46 samples/sec
    INFO:root:Iteration[37] Train-accuracy=0.875900
    INFO:root:Iteration[37] Time cost=355.997
    INFO:root:Iteration[37] Validation-accuracy=0.681330
    INFO:root:Saved checkpoint to "model/cifar_100-0038.params"
    INFO:root:Batch [200]	Speed: 141.09 samples/sec
    INFO:root:Batch [400]	Speed: 140.45 samples/sec
    INFO:root:Batch [600]	Speed: 140.45 samples/sec
    INFO:root:Iteration[38] Train-accuracy=0.879902
    INFO:root:Iteration[38] Time cost=355.928
    INFO:root:Iteration[38] Validation-accuracy=0.688694
    INFO:root:Saved checkpoint to "model/cifar_100-0039.params"


Without reducing learning rate, this model is able to achieve state-of-art result.

Let's reduce learning rate to train a few more rounds.



```python
# load params from saved model
num_epoch = 38
model_prefix = "model/cifar_100"
tmp_model = mx.model.FeedForward.load(model_prefix, epoch)

# create new model with params
num_epoch = 6
model_prefix = "model/cifar_100_stage2"
model = mx.model.FeedForward(ctx=mx.gpu(), symbol=softmax, num_epoch=num_epoch,
                             learning_rate=0.01, momentum=0.9, wd=0.0001,
                             arg_params=tmp_model.arg_params, aux_params=tmp_model.aux_params,)


model.fit(X=train_dataiter,
          eval_data=test_dataiter,
          eval_metric="accuracy",
          batch_end_callback=mx.callback.Speedometer(batch_size, 200),
          epoch_end_callback=mx.callback.do_checkpoint(model_prefix))
```

    INFO:root:Start training with [gpu(0)]
    INFO:root:Batch [200]	Speed: 147.84 samples/sec
    INFO:root:Batch [400]	Speed: 139.77 samples/sec
    INFO:root:Batch [600]	Speed: 140.17 samples/sec
    INFO:root:Iteration[0] Train-accuracy=0.951866
    INFO:root:Iteration[0] Time cost=353.261
    INFO:root:Iteration[0] Validation-accuracy=0.744924
    INFO:root:Saved checkpoint to "model/cifar_100_stage2-0001.params"
    INFO:root:Batch [200]	Speed: 141.02 samples/sec
    INFO:root:Batch [400]	Speed: 140.35 samples/sec
    INFO:root:Batch [600]	Speed: 140.39 samples/sec
    INFO:root:Iteration[1] Train-accuracy=0.976012
    INFO:root:Iteration[1] Time cost=356.142
    INFO:root:Iteration[1] Validation-accuracy=0.747213
    INFO:root:Saved checkpoint to "model/cifar_100_stage2-0002.params"
    INFO:root:Batch [200]	Speed: 140.77 samples/sec
    INFO:root:Batch [400]	Speed: 140.49 samples/sec
    INFO:root:Batch [600]	Speed: 139.74 samples/sec
    INFO:root:Iteration[2] Train-accuracy=0.983335
    INFO:root:Iteration[2] Time cost=356.680
    INFO:root:Iteration[2] Validation-accuracy=0.746716
    INFO:root:Saved checkpoint to "model/cifar_100_stage2-0003.params"
    INFO:root:Batch [200]	Speed: 140.64 samples/sec
    INFO:root:Batch [400]	Speed: 140.16 samples/sec
    INFO:root:Batch [600]	Speed: 140.17 samples/sec
    INFO:root:Iteration[3] Train-accuracy=0.987076
    INFO:root:Iteration[3] Time cost=356.758
    INFO:root:Iteration[3] Validation-accuracy=0.755971
    INFO:root:Saved checkpoint to "model/cifar_100_stage2-0004.params"
    INFO:root:Batch [200]	Speed: 140.58 samples/sec
    INFO:root:Batch [400]	Speed: 139.97 samples/sec
    INFO:root:Batch [600]	Speed: 139.89 samples/sec
    INFO:root:Iteration[4] Train-accuracy=0.989850
    INFO:root:Iteration[4] Time cost=358.025
    INFO:root:Iteration[4] Validation-accuracy=0.752090
    INFO:root:Saved checkpoint to "model/cifar_100_stage2-0005.params"
    INFO:root:Batch [200]	Speed: 140.18 samples/sec
    INFO:root:Batch [400]	Speed: 139.61 samples/sec
    INFO:root:Batch [600]	Speed: 139.32 samples/sec
    INFO:root:Iteration[5] Train-accuracy=0.991037
    INFO:root:Iteration[5] Time cost=358.366
    INFO:root:Iteration[5] Validation-accuracy=0.752886
    INFO:root:Saved checkpoint to "model/cifar_100_stage2-0006.params"
    INFO:root:Batch [200]	Speed: 140.42 samples/sec
    INFO:root:Batch [400]	Speed: 139.11 samples/sec
    INFO:root:Batch [600]	Speed: 139.29 samples/sec
    INFO:root:Iteration[6] Train-accuracy=0.992858
    INFO:root:Iteration[6] Time cost=358.961
    INFO:root:Iteration[6] Validation-accuracy=0.756867
    INFO:root:Saved checkpoint to "model/cifar_100_stage2-0007.params"




**Reference**

[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).

[2] Xu, Bing, et al. "Empirical Evaluation of Rectified Activations in Convolutional Network." arXiv preprint arXiv:1505.00853 (2015).

[3] Srivastava, Rupesh Kumar, Klaus Greff, and Jürgen Schmidhuber. "Highway Networks." arXiv preprint arXiv:1505.00387 (2015).

[4] Lee, Chen-Yu, et al. "Deeply-supervised nets." arXiv preprint arXiv:1409.5185 (2014).


```python

```
