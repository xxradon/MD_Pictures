
# Composite symbols into component
In this example we will show how to make an Inception network by forming single symbol into component.

Inception is currently best model. Compared to other models, it has much less parameters, and with best performance. However, it is much more complex than sequence feedforward network.

The Inception network in this example is refer to ```Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).```




```python
import mxnet as mx
```

For complex network such as inception network, building from single symbol is painful, we can make simple ```ComponentFactory``` to simplfiy the procedure.

Except difference in number of filter, we find 2 major differences in each Inception module, so we can build two factories plus one basic ```Convolution + BatchNorm + ReLU``` factory to simplfiy the problem.



```python
# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act
```

We can visualize our basic component


```python
prev = mx.symbol.Variable(name="Previos Output")
conv_comp = ConvFactory(data=prev, num_filter=64, kernel=(7,7), stride=(2, 2))
mx.viz.plot_network(symbol=conv_comp)
```




![svg](output_5_0.svg)



The next step is making a component factory with all ```stride=(1, 1)```


```python
# param mapping to paper:
# num_1x1      >>>  #1x1
# num_3x3red   >>>  #3x3 reduce
# num_3x3      >>>  #3x3
# num_d3x3red  >>>  double #3x3 reduce
# num_d3x3     >>>  double #3x3
# pool         >>>  Pool
# proj         >>>  proj
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
```


```python
# We can also visualize network with feature map shape information
# In this case, we must provide all necessary input shape info as a dict
prev = mx.symbol.Variable(name="Previos Output")
in3a = InceptionFactoryA(prev, 64, 64, 64, 64, 96, "avg", 32, name="in3a")
# shape info
# Note shape info must contain batch size although we ignore batch size in graph to save space
batch_size = 128
shape = {"Previos Output" : (batch_size, 3, 28, 28)}
# plot
mx.viz.plot_network(symbol=in3a, shape=shape)
```




![svg](output_8_0.svg)



We will make the other factory with ```stride=(2, 2)```


```python
# param mapping to paper:
# num_1x1      >>>  #1x1 (not exist!)
# num_3x3red   >>>  #3x3 reduce
# num_3x3      >>>  #3x3
# num_d3x3red  >>>  double #3x3 reduce
# num_d3x3     >>>  double #3x3
# pool         >>>  Pool (not needed, all are max pooling)
# proj         >>>  proj (not exist!)
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


```python
prev = mx.symbol.Variable(name="Previos Output")
in3c = InceptionFactoryB(prev, 128, 160, 64, 96, name='in3c')
mx.viz.plot_network(symbol=in3c)
```




![svg](output_11_0.svg)



Now we can use these factories to build the whole network


```python
# data
data = mx.symbol.Variable(name="data")
# stage 1
conv1 = ConvFactory(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name='conv1')
pool1 = mx.symbol.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), name='pool1', pool_type='max')
# stage 2
conv2red = ConvFactory(data=pool1, num_filter=64, kernel=(1, 1), stride=(1, 1), name='conv2red')
conv2 = ConvFactory(data=conv2red, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv2')
pool2 = mx.symbol.Pooling(data=conv2, kernel=(3, 3), stride=(2, 2), name='pool2', pool_type='max')
# stage 3
in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, "avg", 32, name='in3a')
in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, name='in3b')
in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, name='in3c')
# stage 4
in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, name='in4a')
in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, name='in4b')
in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, name='in4c')
in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, name='in4d')
in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, name='in4e')
# stage 5
in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, name='in5a')
in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, name='in5b')
# global avg pooling
avg = mx.symbol.Pooling(data=in5b, kernel=(7, 7), stride=(1, 1), name="global_avg", pool_type='avg')
# linear classifier
flatten = mx.symbol.Flatten(data=avg)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=1000)
softmax = mx.symbol.SoftmaxOutput(data=fc1)

# if you like, you can visualize full network structure
mx.viz.plot_network(symbol=softmax, shape={"data" : (128, 3, 224, 224)})
```




![svg](output_13_0.svg)




```python

```
