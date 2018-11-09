
# Source Code studies: Implementation of AlexNet in Pytorch
Learn Pytorch internals from its implementation of AlexNet

PURPOSE: PRIVATE WORKING NOTES

The plan is to 
* walk through all the layers: from AlexNet python class to cuDNN (or low layer CPU) functions.
* see where the backend layers (CPU/GPU) are set; where is the correct place to put, say, ARM-based backend


Links:

"A Walk-through of AlexNet" by Hao Gao in Medium [link](https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637)

AlexNet class in PyTorch is defined [link](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)

"A Tour of PyTorch Internals (Part I)" is [link](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)

"PyTorch Internals" (Part II) - The Build System" is [link](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/)

PyTorch issue: "Implement similar PyTorch function as model.summary() in keras?" is [here](https://github.com/pytorch/pytorch/issues/2001)

Stackoverflow: "What's the best way to generate a UML diagram from Python source code?" [here](https://stackoverflow.com/questions/260165/whats-the-best-way-to-generate-a-uml-diagram-from-python-source-code)



## class AlexNet
_Everything is a Module_. 

AlexNet itself and all its defining elements inherit from the class Module.<br>


`Sequential<-Module`<br>
`Conv2d<-_ConvNd<-Module`<br>
`ReLU<-Threshold<-Module`<br>
`MaxPool2d<-_MaxPoolNd<-Module`<br><br>
`Dropout<-_DropoutNd<-Module`<br>
`Linear<-Module`<br>

![PyTorch nn classes making AlexNet](imgs/AlexNet_class_hierarchy.bmp "PyTorch nn classes making AlexNet")

## class nn.Module

"Modules can also contain other Modules, allowing to nest them in a tree structure"

### nn.Module members

``` python
    def __init__(self):
        self._backend = thnn_backend
        
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        
        self._modules = OrderedDict() #nesting support
        
        self.training = True
```

### nn.Module methods:

``` python
forward(self, *input) #rases NotImplementedError in the Module; Should be overridden by all subclasses

#"adders":
register_buffer(self, name, tensor)
register_parameter(self, name, param)
add_module(self, name, module)

apply(self, fn) #Applies ``fn`` recursively to every submodule

#"movers" to from gpu/cpu
cuda(self, device=None)
cpu(self)

#"type transformers:"
type(self, dst_type) #Casts all parameters and buffers to :attr:`dst_type`
float(self)
double(self)
half(self)

#move and cast
to(self, *args, **kwargs)


__call__(self, *input, **kwargs)

state_dict(self, destination=None, prefix='', keep_vars=False)

#iterators:
parameters(self, recurse=True)
named_parameters(self, prefix='', recurse=True)
buffers(self, recurse=True)
named_buffers(self, prefix='', recurse=True)
children(self)
named_children(self)
modules(self)
named_modules(self, memo=None, prefix='')

#mode setting
train(self, mode=True)
eval(self)

zero_grad(self)

```
