
# CIFAR-10 Recipe
In this notebook, we will show how to train a state-of-art CIFAR-10 network with MXNet and extract feature from the network.
This example wiil cover

- Network/Data definition 
- Multi GPU training
- Model saving and loading
- Prediction/Extracting Feature



```python
import mxnet as mx
import logging
import numpy as np

# setup logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
```

First, let's make some helper function to let us build a simplified Inception Network. More details about how to composite symbol into component can be found at [composite_symbol](composite_symbol.ipynb)


```python
# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu"):
    # there is an optional parameter ```wrokshpace``` may influece convolution performance
    # default, the workspace is set to 256(MB)
    # you may set larger value, but convolution layer only requires its needed but not exactly
    # MXNet will handle reuse of workspace without parallelism conflict
    conv = mx.symbol.Convolution(data=data, workspace=256,
                                 num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type=act_type)
    return act
```


```python
# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1))
    # pool
    pool = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max')
    # concat
    concat = mx.symbol.Concat(*[conv, pool])
    return concat
```


```python
# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3)
    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3])
    return concat
```

Now we can build a network with these component factories


```python
data = mx.symbol.Variable(name="data")
conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu")
in3a = SimpleFactory(conv1, 32, 32)
in3b = SimpleFactory(in3a, 32, 48)
in3c = DownsampleFactory(in3b, 80)
in4a = SimpleFactory(in3c, 112, 48)
in4b = SimpleFactory(in4a, 96, 64)
in4c = SimpleFactory(in4b, 80, 80)
in4d = SimpleFactory(in4c, 48, 96)
in4e = DownsampleFactory(in4d, 96)
in5a = SimpleFactory(in4e, 176, 160)
in5b = SimpleFactory(in5a, 176, 160)
pool = mx.symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_avg")
flatten = mx.symbol.Flatten(data=pool)
fc = mx.symbol.FullyConnected(data=flatten, num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(name='softmax',data=fc)
```


```python
# If you'd like to see the network structure, run the plot_network function
#mx.viz.plot_network(symbol=softmax,node_attrs={'shape':'oval','fixedsize':'false'}) 
```


```python
# We will make model with current current symbol
# For demo purpose, this model only train 1 epoch
# We will use the first GPU to do training
num_epoch = 1
model = mx.model.FeedForward(ctx=mx.gpu(), symbol=softmax, num_epoch=num_epoch,
                             learning_rate=0.05, momentum=0.9, wd=0.00001)

# we can add learning rate scheduler to the model
# model = mx.model.FeedForward(ctx=mx.gpu(), symbol=softmax, num_epoch=num_epoch,
#                              learning_rate=0.05, momentum=0.9, wd=0.00001,
#                              lr_scheduler=mx.misc.FactorScheduler(2))
# In this example. learning rate will be reduced to 0.1 * previous learning rate for every 2 epochs
```

If we have multiple GPU, for eaxmple, 4 GPU, we can utilize them without any difficulty


```python
# num_devs = 4
# model = mx.model.FeedForward(ctx=[mx.gpu(i) for i in range(num_devs)], symbol=softmax, num_epoch = 1,
#                              learning_rate=0.05, momentum=0.9, wd=0.00001)
```

Next step is declaring data iterator. The original CIFAR-10 data is 3x32x32 in binary format, we provides RecordIO format, so we can use Image RecordIO format. For more infomation about Image RecordIO Iterator, check [document](https://mxnet.readthedocs.org/en/latest/python/io.html).


```python
# Use utility function in test to download the data
# or manualy prepar
import sys
sys.path.append("../../tests/python/common") # change the path to mxnet's tests/
import get_data
get_data.GetCifar10()
# After we get the data, we can declare our data iterator
# The iterator will automatically create mean image file if it doesn't exist
batch_size = 128
total_batch = 50000 / 128 + 1
# Train iterator make batch of 128 image, and random crop each image into 3x28x28 from original 3x32x32
train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=True,
        rand_mirror=True,
        data_shape=(3,28,28),
        batch_size=batch_size,
        preprocess_threads=1)
# test iterator make batch of 128 image, and center crop each image into 3x28x28 from original 3x32x32
# Note: We don't need round batch in test because we only test once at one time
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,28,28),
        batch_size=batch_size,
        round_batch=False,
        preprocess_threads=1)
```

Now we can fit the model with data. 


```python
model.fit(X=train_dataiter,
          eval_data=test_dataiter,
          eval_metric="accuracy",
          batch_end_callback=mx.callback.Speedometer(batch_size))

# if we want to save model after every epoch, we can add check_point call back
# model_prefix = './cifar_'
# model.fit(X=train_dataiter,
#           eval_data=test_dataiter,
#           eval_metric="accuracy",
#           batch_end_callback=mx.helper.Speedometer(batch_size),
#           epoch_end_callback=mx.callback.do_checkpoint(model_prefix))

```

    INFO:root:Start training with [gpu(0)]
    INFO:root:Iter[0] Batch [50]	Speed: 1053.96 samples/sec
    INFO:root:Iter[0] Batch [100]	Speed: 1021.90 samples/sec
    INFO:root:Iter[0] Batch [150]	Speed: 1020.08 samples/sec
    INFO:root:Iter[0] Batch [200]	Speed: 1017.71 samples/sec
    INFO:root:Iter[0] Batch [250]	Speed: 1008.16 samples/sec
    INFO:root:Iter[0] Batch [300]	Speed: 1011.40 samples/sec
    INFO:root:Iter[0] Batch [350]	Speed: 995.93 samples/sec
    INFO:root:Epoch[0] Train-accuracy=0.719769
    INFO:root:Epoch[0] Time cost=50.322
    INFO:root:Epoch[0] Validation-accuracy=0.660008


After only 1 epoch, our model is able to acheive about 65% accuracy on testset(If not, try more times).
We can save our model by calling either ```save``` or using ```pickle```.



```python
# using pickle
import pickle
smodel = pickle.dumps(model)
# using saving (recommended)
# We get the benefit being able to directly load/save from cloud storage(S3, HDFS)
prefix = "cifar10"
model.save(prefix)
```

    INFO:root:Saved checkpoint to "cifar10-0001.params"


To load saved model, you can use ```pickle``` if the model is generated by ```pickle```, or use ```load``` if it is generated by ```save```


```python
# use pickle
model2 = pickle.loads(smodel)
# using load method (able to load from S3/HDFS directly)
model3 = mx.model.FeedForward.load(prefix, num_epoch, ctx=mx.gpu())
```

We can use the model to do prediction


```python
prob = model3.predict(test_dataiter)
logging.info('Finish predict...')
# Check the accuracy from prediction
test_dataiter.reset()
# get label
# Because the iterator pad each batch same shape, we want to remove paded samples here

y_batch = []
for dbatch in test_dataiter:
    label = dbatch.label[0].asnumpy()
    pad = test_dataiter.getpad()
    real_size = label.shape[0] - pad
    y_batch.append(label[0:real_size])
y = np.concatenate(y_batch)

# get prediction label from 
py = np.argmax(prob, axis=1)
acc1 = float(np.sum(py == y)) / len(y)
logging.info('final accuracy = %f', acc1)
```

    INFO:root:Finish predict...
    INFO:root:final accuracy = 0.659900


From any symbol, we are able to know its internal feature_maps and bind a new model to extract that feature map


```python
# Predict internal featuremaps
# From a symbol, we are able to get all internals. Note it is still a symbol
internals = softmax.get_internals()
# We get get an internal symbol for the feature.
# By default, the symbol is named as "symbol_name + _output"
# in this case we'd like to get global_avg" layer's output as feature, so its "global_avg_output"
# You may call ```internals.list_outputs()``` to find the target
# but we strongly suggests set a special name for special symbol 
fea_symbol = internals["global_avg_output"]

# Make a new model by using an internal symbol. We can reuse all parameters from model we trained before
# In this case, we must set ```allow_extra_params``` to True 
# Because we don't need params of FullyConnected Layer

feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, 
                                         arg_params=model.arg_params,
                                         aux_params=model.aux_params,
                                         allow_extra_params=True)
# Predict as normal
global_pooling_feature = feature_extractor.predict(test_dataiter)
print(global_pooling_feature.shape)
```

    (10000, 336, 1, 1)



```python

```
