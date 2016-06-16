
# MXNet Tutorial and Hand Written Digit Recognition

In this tutorial we will go through the basic use case of MXNet and also touch on some advanced usages. This example is based on the MNIST dataset, which contains 70,000 images of hand written characters with 28-by-28 pixel size.

This tutorial covers the following topics:
- network definition.
- Variable naming.
- Basic data loading and training with feed-forward deep neural networks.
- Monitoring intermediate outputs for debuging.
- Custom training loop for advanced models.

First let's import the modules and setup logging:


```python
%matplotlib inline
import mxnet as mx
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
```

## Network Definition
Now we can start constructing our network:


```python
# Variables are place holders for input arrays. We give each variable a unique name.
data = mx.symbol.Variable('data')

# The input is fed to a fully connected layer that computes Y=WX+b.
# This is the main computation module in the network.
# Each layer also needs an unique name. We'll talk more about naming in the next section.
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
# Activation layers apply a non-linear function on the previous layer's output.
# Here we use Rectified Linear Unit (ReLU) that computes Y = max(X, 0).
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")

fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")

fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
# Finally we have a loss layer that compares the network's output with label and generates gradient signals.
mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
```

We can visualize the network we just defined with MXNet's visualization module:


```python
mx.viz.plot_network(mlp)
```




![svg](output_6_0.svg)



## Variable Naming

MXNet requires variable names to follow certain conventions:
- All input arrays have a name. This includes inputs (data & label) and model parameters (weight, bias, etc).
- Arrays can be renamed by creating named variable. Otherwise, a default name is given as 'SymbolName_ArrayName'. For example, FullyConnected symbol fc1's weight array is named as 'fc1_weight'.
- Although you can also rename weight arrays with variables, weight array's name should always end with '_weight' and bias array '_bias'. MXNet relies on the suffixes of array names to correctly initialize & update them.

Call list_arguments method on a symbol to get the names of all its inputs:


```python
mlp.list_arguments()
```




    ['data',
     'fc1_weight',
     'fc1_bias',
     'fc2_weight',
     'fc2_bias',
     'fc3_weight',
     'fc3_bias',
     'softmax_label']



## Data Loading

We fetch and load the MNIST dataset and partition it into two sets: 60000 examples for training and 10000 examples for testing. We also visualize a few examples to get an idea of what the dataset looks like.


```python
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist.data.shape[0])
X = mnist.data[p]
Y = mnist.target[p]

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X[i].reshape((28,28)), cmap='Greys_r')
    plt.axis('off')
plt.show()

X = X.astype(np.float32)/255
X_train = X[:60000]
X_test = X[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]
```


![png](output_10_0.png)


Now we can create data iterators from our MNIST data. A data iterator returns a batch of data examples each time for the network to process. MXNet provide a suite of basic DataIters for parsing different data format. Here we use NDArrayIter, which wraps around a numpy array and each time slice a chunk from it along the first dimension.


```python
batch_size = 100
train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)
test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size)
```

## Training

With the network and data source defined, we can finally start to train our model. We do this with MXNet's convenience wrapper for feed forward neural networks (it can also be made to handle RNNs with explicit unrolling). 


```python
model = mx.model.FeedForward(
    ctx = mx.gpu(0),      # Run on GPU 0
    symbol = mlp,         # Use the network we just defined
    num_epoch = 10,       # Train for 10 epochs
    learning_rate = 0.1,  # Learning rate
    momentum = 0.9,       # Momentum for SGD with momentum
    wd = 0.00001)         # Weight decay for regularization
model.fit(
    X=train_iter,  # Training data set
    eval_data=test_iter,  # Testing data set. MXNet computes scores on test set every epoch
    batch_end_callback = mx.callback.Speedometer(batch_size, 200))  # Logging module to print out progress
```

    INFO:root:Start training with [gpu(0)]
    INFO:root:Epoch[0] Batch [200]	Speed: 70941.64 samples/sec	Train-accuracy=0.389050
    INFO:root:Epoch[0] Batch [400]	Speed: 97857.94 samples/sec	Train-accuracy=0.646450
    INFO:root:Epoch[0] Batch [600]	Speed: 70507.97 samples/sec	Train-accuracy=0.743333
    INFO:root:Epoch[0] Resetting Data Iterator
    INFO:root:Epoch[0] Train-accuracy=0.743333
    INFO:root:Epoch[0] Time cost=1.069
    INFO:root:Epoch[0] Validation-accuracy=0.950800
    INFO:root:Epoch[1] Batch [200]	Speed: 79912.66 samples/sec	Train-accuracy=0.947300
    INFO:root:Epoch[1] Batch [400]	Speed: 58822.31 samples/sec	Train-accuracy=0.954425
    INFO:root:Epoch[1] Batch [600]	Speed: 67124.87 samples/sec	Train-accuracy=0.957733
    INFO:root:Epoch[1] Resetting Data Iterator
    INFO:root:Epoch[1] Train-accuracy=0.957733
    INFO:root:Epoch[1] Time cost=0.893
    INFO:root:Epoch[1] Validation-accuracy=0.959400
    INFO:root:Epoch[2] Batch [200]	Speed: 87015.23 samples/sec	Train-accuracy=0.964450
    INFO:root:Epoch[2] Batch [400]	Speed: 91101.30 samples/sec	Train-accuracy=0.968875
    INFO:root:Epoch[2] Batch [600]	Speed: 88963.21 samples/sec	Train-accuracy=0.970017
    INFO:root:Epoch[2] Resetting Data Iterator
    INFO:root:Epoch[2] Train-accuracy=0.970017
    INFO:root:Epoch[2] Time cost=0.678
    INFO:root:Epoch[2] Validation-accuracy=0.963000
    INFO:root:Epoch[3] Batch [200]	Speed: 66986.68 samples/sec	Train-accuracy=0.973750
    INFO:root:Epoch[3] Batch [400]	Speed: 65680.34 samples/sec	Train-accuracy=0.976575
    INFO:root:Epoch[3] Batch [600]	Speed: 91931.16 samples/sec	Train-accuracy=0.977050
    INFO:root:Epoch[3] Resetting Data Iterator
    INFO:root:Epoch[3] Train-accuracy=0.977050
    INFO:root:Epoch[3] Time cost=0.825
    INFO:root:Epoch[3] Validation-accuracy=0.968000
    INFO:root:Epoch[4] Batch [200]	Speed: 73709.59 samples/sec	Train-accuracy=0.978950
    INFO:root:Epoch[4] Batch [400]	Speed: 85750.82 samples/sec	Train-accuracy=0.980425
    INFO:root:Epoch[4] Batch [600]	Speed: 87061.38 samples/sec	Train-accuracy=0.981183
    INFO:root:Epoch[4] Resetting Data Iterator
    INFO:root:Epoch[4] Train-accuracy=0.981183
    INFO:root:Epoch[4] Time cost=0.739
    INFO:root:Epoch[4] Validation-accuracy=0.967600
    INFO:root:Epoch[5] Batch [200]	Speed: 85031.11 samples/sec	Train-accuracy=0.981950
    INFO:root:Epoch[5] Batch [400]	Speed: 94063.25 samples/sec	Train-accuracy=0.983475
    INFO:root:Epoch[5] Batch [600]	Speed: 97417.46 samples/sec	Train-accuracy=0.984183
    INFO:root:Epoch[5] Resetting Data Iterator
    INFO:root:Epoch[5] Train-accuracy=0.984183
    INFO:root:Epoch[5] Time cost=0.657
    INFO:root:Epoch[5] Validation-accuracy=0.972000
    INFO:root:Epoch[6] Batch [200]	Speed: 96185.84 samples/sec	Train-accuracy=0.984650
    INFO:root:Epoch[6] Batch [400]	Speed: 95023.61 samples/sec	Train-accuracy=0.985850
    INFO:root:Epoch[6] Batch [600]	Speed: 97022.32 samples/sec	Train-accuracy=0.986683
    INFO:root:Epoch[6] Resetting Data Iterator
    INFO:root:Epoch[6] Train-accuracy=0.986683
    INFO:root:Epoch[6] Time cost=0.628
    INFO:root:Epoch[6] Validation-accuracy=0.971900
    INFO:root:Epoch[7] Batch [200]	Speed: 84764.84 samples/sec	Train-accuracy=0.986350
    INFO:root:Epoch[7] Batch [400]	Speed: 87358.40 samples/sec	Train-accuracy=0.986425
    INFO:root:Epoch[7] Batch [600]	Speed: 74520.63 samples/sec	Train-accuracy=0.986517
    INFO:root:Epoch[7] Resetting Data Iterator
    INFO:root:Epoch[7] Train-accuracy=0.986517
    INFO:root:Epoch[7] Time cost=0.737
    INFO:root:Epoch[7] Validation-accuracy=0.973700
    INFO:root:Epoch[8] Batch [200]	Speed: 91634.21 samples/sec	Train-accuracy=0.987450
    INFO:root:Epoch[8] Batch [400]	Speed: 94328.96 samples/sec	Train-accuracy=0.987250
    INFO:root:Epoch[8] Batch [600]	Speed: 91991.24 samples/sec	Train-accuracy=0.987850
    INFO:root:Epoch[8] Resetting Data Iterator
    INFO:root:Epoch[8] Train-accuracy=0.987850
    INFO:root:Epoch[8] Time cost=0.652
    INFO:root:Epoch[8] Validation-accuracy=0.976800
    INFO:root:Epoch[9] Batch [200]	Speed: 66583.86 samples/sec	Train-accuracy=0.986800
    INFO:root:Epoch[9] Batch [400]	Speed: 67393.86 samples/sec	Train-accuracy=0.987500
    INFO:root:Epoch[9] Batch [600]	Speed: 65748.40 samples/sec	Train-accuracy=0.987900
    INFO:root:Epoch[9] Resetting Data Iterator
    INFO:root:Epoch[9] Train-accuracy=0.987900
    INFO:root:Epoch[9] Time cost=0.906
    INFO:root:Epoch[9] Validation-accuracy=0.973800


## Evaluation

After the model is trained, we can evaluate it on a held out test set.
First, lets classity a sample image:


```python
plt.imshow((X_test[0].reshape((28,28))*255).astype(np.uint8), cmap='Greys_r')
plt.show()
print 'Result:', model.predict(X_test[0:1])[0].argmax()
```


![png](output_16_0.png)


    Result: 7


We can also evaluate the model's accuracy on the entire test set:


```python
print 'Accuracy:', model.score(test_iter)*100, '%'
```

    Accuracy: 97.38 %


Now, try if your model recognizes your own hand writing.

Write a digit from 0 to 9 in the box below. Try to put your digit in the middle of the box.


```python
# run hand drawing test
from IPython.display import HTML

def classify(img):
    img = img[len('data:image/png;base64,'):].decode('base64')
    img = cv2.imdecode(np.fromstring(img, np.uint8), -1)
    img = cv2.resize(img[:,:,3], (28,28))
    img = img.astype(np.float32).reshape((1, 784))/255.0
    return model.predict(img)[0].argmax()

html = """<style type="text/css">canvas { border: 1px solid black; }</style><div id="board"><canvas id="myCanvas" width="100px" height="100px">Sorry, your browser doesn't support canvas technology.</canvas><p><button id="classify" onclick="classify()">Classify</button><button id="clear" onclick="myClear()">Clear</button>Result: <input type="text" id="result_output" size="5" value=""></p></div>"""
script = """<script type="text/JavaScript" src="https://ajax.googleapis.com/ajax/libs/jquery/1.4.2/jquery.min.js?ver=1.4.2"></script><script type="text/javascript">function init() {var myCanvas = document.getElementById("myCanvas");var curColor = $('#selectColor option:selected').val();if(myCanvas){var isDown = false;var ctx = myCanvas.getContext("2d");var canvasX, canvasY;ctx.lineWidth = 8;$(myCanvas).mousedown(function(e){isDown = true;ctx.beginPath();var parentOffset = $(this).parent().offset(); canvasX = e.pageX - parentOffset.left;canvasY = e.pageY - parentOffset.top;ctx.moveTo(canvasX, canvasY);}).mousemove(function(e){if(isDown != false) {var parentOffset = $(this).parent().offset(); canvasX = e.pageX - parentOffset.left;canvasY = e.pageY - parentOffset.top;ctx.lineTo(canvasX, canvasY);ctx.strokeStyle = curColor;ctx.stroke();}}).mouseup(function(e){isDown = false;ctx.closePath();});}$('#selectColor').change(function () {curColor = $('#selectColor option:selected').val();});}init();function handle_output(out) {document.getElementById("result_output").value = out.content.data["text/plain"];}function classify() {var kernel = IPython.notebook.kernel;var myCanvas = document.getElementById("myCanvas");data = myCanvas.toDataURL('image/png');document.getElementById("result_output").value = "";kernel.execute("classify('" + data +"')",  { 'iopub' : {'output' : handle_output}}, {silent:false});}function myClear() {var myCanvas = document.getElementById("myCanvas");myCanvas.getContext("2d").clearRect(0, 0, myCanvas.width, myCanvas.height);}</script>"""
HTML(html+script)
```




<style type="text/css">canvas { border: 1px solid black; }</style><div id="board"><canvas id="myCanvas" width="100px" height="100px">Sorry, your browser doesn't support canvas technology.</canvas><p><button id="classify" onclick="classify()">Classify</button><button id="clear" onclick="myClear()">Clear</button>Result: <input type="text" id="result_output" size="5" value=""></p></div><script type="text/JavaScript" src="https://ajax.googleapis.com/ajax/libs/jquery/1.4.2/jquery.min.js?ver=1.4.2"></script><script type="text/javascript">function init() {var myCanvas = document.getElementById("myCanvas");var curColor = $('#selectColor option:selected').val();if(myCanvas){var isDown = false;var ctx = myCanvas.getContext("2d");var canvasX, canvasY;ctx.lineWidth = 8;$(myCanvas).mousedown(function(e){isDown = true;ctx.beginPath();var parentOffset = $(this).parent().offset(); canvasX = e.pageX - parentOffset.left;canvasY = e.pageY - parentOffset.top;ctx.moveTo(canvasX, canvasY);}).mousemove(function(e){if(isDown != false) {var parentOffset = $(this).parent().offset(); canvasX = e.pageX - parentOffset.left;canvasY = e.pageY - parentOffset.top;ctx.lineTo(canvasX, canvasY);ctx.strokeStyle = curColor;ctx.stroke();}}).mouseup(function(e){isDown = false;ctx.closePath();});}$('#selectColor').change(function () {curColor = $('#selectColor option:selected').val();});}init();function handle_output(out) {document.getElementById("result_output").value = out.content.data["text/plain"];}function classify() {var kernel = IPython.notebook.kernel;var myCanvas = document.getElementById("myCanvas");data = myCanvas.toDataURL('image/png');document.getElementById("result_output").value = "";kernel.execute("classify('" + data +"')",  { 'iopub' : {'output' : handle_output}}, {silent:false});}function myClear() {var myCanvas = document.getElementById("myCanvas");myCanvas.getContext("2d").clearRect(0, 0, myCanvas.width, myCanvas.height);}</script>



## Debugging

DNNs can perform poorly for a lot of reasons, like learning rate too big/small, initialization too big/small, network structure not reasonable, etc. When this happens it's often helpful to print out the weights and intermediate outputs to understand what's going on. MXNet provides a monitor utility that does this:


```python
def norm_stat(d):
    """The statistics you want to see.
    We compute the L2 norm here but you can change it to anything you like."""
    return mx.nd.norm(d)/np.sqrt(d.size)
mon = mx.mon.Monitor(
    100,                 # Print every 100 batches
    norm_stat,           # The statistics function defined above
    pattern='.*weight',  # A regular expression. Only arrays with name matching this pattern will be included.
    sort=True)           # Sort output by name
model = mx.model.FeedForward(ctx = mx.gpu(0), symbol = mlp, num_epoch = 1,
                             learning_rate = 0.1, momentum = 0.9, wd = 0.00001)
model.fit(X=train_iter, eval_data=test_iter, monitor=mon,  # Set the monitor here
          batch_end_callback = mx.callback.Speedometer(100, 100))
```

    INFO:root:Start training with [gpu(0)]
    INFO:root:Batch:       1 fc1_backward_weight            0.000519617	
    INFO:root:Batch:       1 fc1_weight                     0.00577777	
    INFO:root:Batch:       1 fc2_backward_weight            0.00164324	
    INFO:root:Batch:       1 fc2_weight                     0.00577121	
    INFO:root:Batch:       1 fc3_backward_weight            0.00490826	
    INFO:root:Batch:       1 fc3_weight                     0.00581168	
    INFO:root:Epoch[0] Batch [100]	Speed: 56125.81 samples/sec	Train-accuracy=0.141400
    INFO:root:Batch:     101 fc1_backward_weight            0.170696	
    INFO:root:Batch:     101 fc1_weight                     0.0077417	
    INFO:root:Batch:     101 fc2_backward_weight            0.300237	
    INFO:root:Batch:     101 fc2_weight                     0.0188219	
    INFO:root:Batch:     101 fc3_backward_weight            1.26234	
    INFO:root:Batch:     101 fc3_weight                     0.0678799	
    INFO:root:Epoch[0] Batch [200]	Speed: 76573.19 samples/sec	Train-accuracy=0.419000
    INFO:root:Batch:     201 fc1_backward_weight            0.224993	
    INFO:root:Batch:     201 fc1_weight                     0.0224456	
    INFO:root:Batch:     201 fc2_backward_weight            0.574649	
    INFO:root:Batch:     201 fc2_weight                     0.0481841	
    INFO:root:Batch:     201 fc3_backward_weight            1.50356	
    INFO:root:Batch:     201 fc3_weight                     0.223626	
    INFO:root:Epoch[0] Batch [300]	Speed: 82821.98 samples/sec	Train-accuracy=0.574900
    INFO:root:Batch:     301 fc1_backward_weight            0.128922	
    INFO:root:Batch:     301 fc1_weight                     0.0297723	
    INFO:root:Batch:     301 fc2_backward_weight            0.25938	
    INFO:root:Batch:     301 fc2_weight                     0.0623646	
    INFO:root:Batch:     301 fc3_backward_weight            0.623773	
    INFO:root:Batch:     301 fc3_weight                     0.243092	
    INFO:root:Epoch[0] Batch [400]	Speed: 81133.86 samples/sec	Train-accuracy=0.662375
    INFO:root:Batch:     401 fc1_backward_weight            0.244692	
    INFO:root:Batch:     401 fc1_weight                     0.0343876	
    INFO:root:Batch:     401 fc2_backward_weight            0.42573	
    INFO:root:Batch:     401 fc2_weight                     0.0708167	
    INFO:root:Batch:     401 fc3_backward_weight            0.813565	
    INFO:root:Batch:     401 fc3_weight                     0.252606	
    INFO:root:Epoch[0] Batch [500]	Speed: 79695.23 samples/sec	Train-accuracy=0.716540
    INFO:root:Batch:     501 fc1_backward_weight            0.208892	
    INFO:root:Batch:     501 fc1_weight                     0.0385131	
    INFO:root:Batch:     501 fc2_backward_weight            0.475372	
    INFO:root:Batch:     501 fc2_weight                     0.0783694	
    INFO:root:Batch:     501 fc3_backward_weight            0.984594	
    INFO:root:Batch:     501 fc3_weight                     0.2605	
    INFO:root:Epoch[0] Batch [600]	Speed: 78154.25 samples/sec	Train-accuracy=0.754600
    INFO:root:Epoch[0] Resetting Data Iterator
    INFO:root:Epoch[0] Train-accuracy=0.754600
    INFO:root:Epoch[0] Time cost=0.831
    INFO:root:Epoch[0] Validation-accuracy=0.953200


## Under the hood: Custom Training Loop

`mx.model.FeedForward` is a convenience wrapper for training standard feed forward networks. What if the model you are working with is more complicated? With MXNet, you can easily control every aspect of training by writing your own training loop.

Neural network training typically has 3 steps: forward, backward (gradient), and update. With custom training loop, you can control the details in each step as while as insert complicated computations in between. You can also connect multiple networks together.


```python
# ==================Binding=====================
# The symbol we created is only a graph description.
# To run it, we first need to allocate memory and create an executor by 'binding' it.
# In order to bind a symbol, we need at least two pieces of information: context and input shapes.
# Context specifies which device the executor runs on, e.g. cpu, GPU0, GPU1, etc.
# Input shapes define the executor's input array dimensions.
# MXNet then run automatic shape inference to determine the dimensions of intermediate and output arrays.

# data iterators defines shapes of its output with provide_data and provide_label property.
input_shapes = dict(train_iter.provide_data+train_iter.provide_label)
print 'input_shapes', input_shapes
# We use simple_bind to let MXNet allocate memory for us.
# You can also allocate memory youself and use bind to pass it to MXNet.
exe = mlp.simple_bind(ctx=mx.gpu(0), **input_shapes)

# ===============Initialization=================
# First we get handle to input arrays
arg_arrays = dict(zip(mlp.list_arguments(), exe.arg_arrays))
data = arg_arrays[train_iter.provide_data[0][0]]
label = arg_arrays[train_iter.provide_label[0][0]]

# We initialize the weights with uniform distribution on (-0.01, 0.01).
init = mx.init.Uniform(scale=0.01)
for name, arr in arg_arrays.items():
    if name not in input_shapes:
        init(name, arr)
    
# We also need to create an optimizer for updating weights
opt = mx.optimizer.SGD(
    learning_rate=0.1,
    momentum=0.9,
    wd=0.00001,
    rescale_grad=1.0/train_iter.batch_size)
updater = mx.optimizer.get_updater(opt)

# Finally we need a metric to print out training progress
metric = mx.metric.Accuracy()

# Training loop begines
for epoch in range(10):
    train_iter.reset()
    metric.reset()
    t = 0
    for batch in train_iter:
        # Copy data to executor input. Note the [:].
        data[:] = batch.data[0]
        label[:] = batch.label[0]
        
        # Forward
        exe.forward(is_train=True)
        
        # You perform operations on exe.outputs here if you need to.
        # For example, you can stack a CRF on top of a neural network.
        
        # Backward
        exe.backward()
        
        # Update
        for i, pair in enumerate(zip(exe.arg_arrays, exe.grad_arrays)):
            weight, grad = pair
            updater(i, grad, weight)
        metric.update(batch.label, exe.outputs)
        t += 1
        if t % 100 == 0:
            print 'epoch:', epoch, 'iter:', t, 'metric:', metric.get()

```

    input_shapes {'softmax_label': (100,), 'data': (100, 784)}
    epoch: 0 iter: 100 metric: ('accuracy', 0.1427)
    epoch: 0 iter: 200 metric: ('accuracy', 0.42695)
    epoch: 0 iter: 300 metric: ('accuracy', 0.5826333333333333)
    epoch: 0 iter: 400 metric: ('accuracy', 0.66875)
    epoch: 0 iter: 500 metric: ('accuracy', 0.72238)
    epoch: 0 iter: 600 metric: ('accuracy', 0.7602166666666667)
    epoch: 1 iter: 100 metric: ('accuracy', 0.9504)
    epoch: 1 iter: 200 metric: ('accuracy', 0.9515)
    epoch: 1 iter: 300 metric: ('accuracy', 0.9547666666666667)
    epoch: 1 iter: 400 metric: ('accuracy', 0.95665)
    epoch: 1 iter: 500 metric: ('accuracy', 0.95794)
    epoch: 1 iter: 600 metric: ('accuracy', 0.95935)
    epoch: 2 iter: 100 metric: ('accuracy', 0.9657)
    epoch: 2 iter: 200 metric: ('accuracy', 0.96715)
    epoch: 2 iter: 300 metric: ('accuracy', 0.9698)
    epoch: 2 iter: 400 metric: ('accuracy', 0.9702)
    epoch: 2 iter: 500 metric: ('accuracy', 0.97104)
    epoch: 2 iter: 600 metric: ('accuracy', 0.9717)
    epoch: 3 iter: 100 metric: ('accuracy', 0.976)
    epoch: 3 iter: 200 metric: ('accuracy', 0.97575)
    epoch: 3 iter: 300 metric: ('accuracy', 0.9772666666666666)
    epoch: 3 iter: 400 metric: ('accuracy', 0.9771)
    epoch: 3 iter: 500 metric: ('accuracy', 0.9771)
    epoch: 3 iter: 600 metric: ('accuracy', 0.97755)
    epoch: 4 iter: 100 metric: ('accuracy', 0.9805)
    epoch: 4 iter: 200 metric: ('accuracy', 0.9803)
    epoch: 4 iter: 300 metric: ('accuracy', 0.9814666666666667)
    epoch: 4 iter: 400 metric: ('accuracy', 0.981175)
    epoch: 4 iter: 500 metric: ('accuracy', 0.98132)
    epoch: 4 iter: 600 metric: ('accuracy', 0.98145)
    epoch: 5 iter: 100 metric: ('accuracy', 0.9837)
    epoch: 5 iter: 200 metric: ('accuracy', 0.98365)
    epoch: 5 iter: 300 metric: ('accuracy', 0.9835333333333334)
    epoch: 5 iter: 400 metric: ('accuracy', 0.98395)
    epoch: 5 iter: 500 metric: ('accuracy', 0.984)
    epoch: 5 iter: 600 metric: ('accuracy', 0.9842166666666666)
    epoch: 6 iter: 100 metric: ('accuracy', 0.9848)
    epoch: 6 iter: 200 metric: ('accuracy', 0.98475)
    epoch: 6 iter: 300 metric: ('accuracy', 0.9858333333333333)
    epoch: 6 iter: 400 metric: ('accuracy', 0.98555)
    epoch: 6 iter: 500 metric: ('accuracy', 0.9855)
    epoch: 6 iter: 600 metric: ('accuracy', 0.9856333333333334)
    epoch: 7 iter: 100 metric: ('accuracy', 0.9842)
    epoch: 7 iter: 200 metric: ('accuracy', 0.98625)
    epoch: 7 iter: 300 metric: ('accuracy', 0.9869)
    epoch: 7 iter: 400 metric: ('accuracy', 0.9877)
    epoch: 7 iter: 500 metric: ('accuracy', 0.98774)
    epoch: 7 iter: 600 metric: ('accuracy', 0.9875333333333334)
    epoch: 8 iter: 100 metric: ('accuracy', 0.9864)
    epoch: 8 iter: 200 metric: ('accuracy', 0.9878)
    epoch: 8 iter: 300 metric: ('accuracy', 0.9886666666666667)
    epoch: 8 iter: 400 metric: ('accuracy', 0.98885)
    epoch: 8 iter: 500 metric: ('accuracy', 0.98918)
    epoch: 8 iter: 600 metric: ('accuracy', 0.9894666666666667)
    epoch: 9 iter: 100 metric: ('accuracy', 0.9884)
    epoch: 9 iter: 200 metric: ('accuracy', 0.98855)
    epoch: 9 iter: 300 metric: ('accuracy', 0.9894666666666667)
    epoch: 9 iter: 400 metric: ('accuracy', 0.98945)
    epoch: 9 iter: 500 metric: ('accuracy', 0.98972)
    epoch: 9 iter: 600 metric: ('accuracy', 0.9899333333333333)


## New Operators

MXNet provides a repository of common operators (or layers). However, new models often require new layers. There are several ways to [create new operators](https://mxnet.readthedocs.org/en/latest/tutorial/new_op_howto.html) with MXNet. Here we talk about the easiest way: pure python. 


```python
# Define custom softmax operator
class NumpySoftmax(mx.operator.NumpyOp):
    def __init__(self):
        # Call the parent class constructor. 
        # Because NumpySoftmax is a loss layer, it doesn't need gradient input from layers above.
        super(NumpySoftmax, self).__init__(need_top_grad=False)
    
    def list_arguments(self):
        # Define the input to NumpySoftmax.
        return ['data', 'label']

    def list_outputs(self):
        # Define the output.
        return ['output']

    def infer_shape(self, in_shape):
        # Calculate the dimensions of the output (and missing inputs) from (some) input shapes.
        data_shape = in_shape[0]  # shape of first argument 'data'
        label_shape = (in_shape[0][0],)  # 'label' should be one dimensional and has batch_size instances.
        output_shape = in_shape[0] # 'output' dimension is the same as the input.
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]  # 'data'
        y = out_data[0]  # 'output'
        
        # Compute softmax
        y[:] = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]  # 'label'
        l = l.reshape((l.size,)).astype(np.int)  # cast to int
        y = out_data[0]  # 'output'
        dx = in_grad[0]  # gradient for 'data'
        
        # Compute gradient
        dx[:] = y
        dx[np.arange(l.shape[0]), l] -= 1.0

numpy_softmax = NumpySoftmax()

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
# Use the new operator we just defined instead of the standard softmax operator.
mlp = numpy_softmax(data=fc3, name = 'softmax')

model = mx.model.FeedForward(ctx = mx.gpu(0), symbol = mlp, num_epoch = 2,
                             learning_rate = 0.1, momentum = 0.9, wd = 0.00001)
model.fit(X=train_iter, eval_data=test_iter,
          batch_end_callback = mx.callback.Speedometer(100, 100))
```

    INFO:root:Start training with [gpu(0)]
    INFO:root:Epoch[0] Batch [100]	Speed: 53975.81 samples/sec	Train-accuracy=0.167800
    INFO:root:Epoch[0] Batch [200]	Speed: 75720.80 samples/sec	Train-accuracy=0.455800
    INFO:root:Epoch[0] Batch [300]	Speed: 73701.82 samples/sec	Train-accuracy=0.602833
    INFO:root:Epoch[0] Batch [400]	Speed: 65162.74 samples/sec	Train-accuracy=0.684375
    INFO:root:Epoch[0] Batch [500]	Speed: 65920.09 samples/sec	Train-accuracy=0.735120
    INFO:root:Epoch[0] Batch [600]	Speed: 67870.31 samples/sec	Train-accuracy=0.770333
    INFO:root:Epoch[0] Resetting Data Iterator
    INFO:root:Epoch[0] Train-accuracy=0.770333
    INFO:root:Epoch[0] Time cost=0.923
    INFO:root:Epoch[0] Validation-accuracy=0.950400
    INFO:root:Epoch[1] Batch [100]	Speed: 54063.96 samples/sec	Train-accuracy=0.946700
    INFO:root:Epoch[1] Batch [200]	Speed: 74701.53 samples/sec	Train-accuracy=0.949700
    INFO:root:Epoch[1] Batch [300]	Speed: 69534.33 samples/sec	Train-accuracy=0.953400
    INFO:root:Epoch[1] Batch [400]	Speed: 76418.05 samples/sec	Train-accuracy=0.954875
    INFO:root:Epoch[1] Batch [500]	Speed: 68825.54 samples/sec	Train-accuracy=0.956340
    INFO:root:Epoch[1] Batch [600]	Speed: 74324.13 samples/sec	Train-accuracy=0.958083
    INFO:root:Epoch[1] Resetting Data Iterator
    INFO:root:Epoch[1] Train-accuracy=0.958083
    INFO:root:Epoch[1] Time cost=0.879
    INFO:root:Epoch[1] Validation-accuracy=0.957200



```python

```
