
# Deep Dive into Implementation of AlexNet in Pytorch
Source Code studies

PURPOSE: PRIVATE WORKING NOTES

### Table of Contents

* [1. Introduction](#section1)
* [2. class AlexNet](#section2)
* [3. class nn.Module](#section3)
    * [3.1 nn.Module members](#section3.1)
    * [3.2 nn.Module methods](#section3.2)
* [4. class nn.Sequence](#section4)
* [5. classs Conv2d and \_ConvNd](#section5)
    * [5.1 Conv2d](#section5.1)

<a id='section1'></a>
## 1. Introduction


The plan is to Learn Pytorch internals from its implementation of AlexNet, to 
* walk through all the layers: from AlexNet python class to cuDNN (or low layer CPU) functions.
* see where the backend layers (CPU/GPU) are set; where is the correct place to put, say, ARM-based backend

AlexNet is selected as an example of a relatively simple convolutional network.

Links:

"Convolutional Neural Networks overview" in cs231n by Andrej Karpathy [link](http://cs231n.github.io/convolutional-networks/)

"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya Sutskever, and 
Geoffrey E. Hinton [link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

"A Walk-through of AlexNet" by Hao Gao in Medium [link](https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637)

AlexNet class in PyTorch is defined [link](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)

"A Tour of PyTorch Internals (Part I)" is [link](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)

"PyTorch Internals" (Part II) - The Build System" is [link](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/)

PyTorch issue: "Implement similar PyTorch function as model.summary() in keras?" is [link](https://github.com/pytorch/pytorch/issues/2001)

Stackoverflow: "What's the best way to generate a UML diagram from Python source code?" [link](https://stackoverflow.com/questions/260165/whats-the-best-way-to-generate-a-uml-diagram-from-python-source-code)

"Convolutions with cuDNN" by 
Peter Goldsborough [link](http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/)

"A Tutorial on Filter Groups (Grouped Convolution)" by Yani Ioannou [link](https://blog.yani.io/filter-group-tutorial/) and the paper "Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups" [link](https://arxiv.org/pdf/1605.06489.pdf)

<a id='section2'></a>
## 2. class AlexNet
_Everything is a Module_. 

AlexNet itself and all its defining elements inherit from the class Module.<br>


`Sequential<-Module`<br>
`Conv2d<-_ConvNd<-Module`<br>
`ReLU<-Threshold<-Module`<br>
`MaxPool2d<-_MaxPoolNd<-Module`<br><br>
`Dropout<-_DropoutNd<-Module`<br>
`Linear<-Module`<br>

![PyTorch nn classes making AlexNet](imgs/AlexNet_class_hierarchy.bmp "PyTorch nn classes making AlexNet")

<a id='section3'></a>
## 3. class nn.Module

nn.Modules can contain other nn.Modules, allowing to nest them in a tree structure

<a id='section3.1'></a>
### 3.1. nn.Module members

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

<a id='section3.2'></a>
### 3.2 nn.Module methods

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


__call__(self, *input, **kwargs) # performace forward and backward passes - TBD verify and get details

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

<a id='section4'></a>
## 4. class nn.Sequence

nn.Sequence constructor runs over the \*args or, alternatively, an ordered dict of modules and for each one of them calls nn.Module's `add_module()` function.

```python
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
```
`add_module()` function adds a module to the `_modules` OrderedDict()

The `forward()` function of nn.Sequence iteratively calls `forward()` function for each sub Module
```python
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
```

<a id='section5'></a>
## 5. classs Conv2d and \_ConvNd

<a id='section5.1'></a>
### 5.1 Conv2d

`Conv2d` "applies a 2D convolution over an input signal composed of several input planes"

`Conv2d` class is very small 
```python
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

```

Initialization of the `Conv2d` class members are based on `_ConvNd` class's capabilities. These capabilities are generic for all `ConvXd` classes

The intereseting part is the `forward()` function is implemented by `F.conv2d()` function that belongs to PyTorch's "Functional interface" defined in `functional.py`

```python
conv1d = _add_docstr(torch.conv1d, ...)
```
here we see calls to "C++" functions (TBD get details of the call meachanism): 

`"_add_docstr"` is mapped to `THPModule_addDocStr` method in C++ component `Module.cpp`:
```C++
static PyMethodDef TorchMethods[] = {
  {"_initExtension",  (PyCFunction)THPModule_initExtension,   METH_O,       nullptr},
  {"_autograd_init",  (PyCFunction)THPAutograd_initExtension, METH_NOARGS,  nullptr},
  {"_add_docstr",     (PyCFunction)THPModule_addDocStr,       METH_VARARGS, nullptr},
  ...
```
`Module.cpp` has nothing in common with `nn.Module` class

torch/scsrc/api/src/nn/modules/conv.cpp:
```C++
Tensor Conv2dImpl::forward(Tensor input) {
  AT_ASSERT(input.ndimension() == 4);

...
  return torch::conv2d(
      input,
      weight,
      bias,
      options.stride_,
      options.padding_,
      options.dilation_,
      options.groups_);
}

```

aten/src/ATen/native/Convolution.cpp:

```C++
at::Tensor conv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntList stride, IntList padding, IntList dilation, int64_t groups) {
  return at::convolution(input, weight, bias, stride, padding, dilation,
                         false, {{0, 0}}, groups);
}

...

at::Tensor convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntList stride, IntList padding, IntList dilation,
    bool transposed, IntList output_padding, int64_t groups) {
  auto& ctx = at::globalContext();
  return at::_convolution(input, weight, bias, stride, padding, dilation,
                          transposed, output_padding, groups,
                          ctx.benchmarkCuDNN(), ctx.deterministicCuDNN(), ctx.userEnabledCuDNN());
}


...
    
    

at::Tensor _convolution(
    const Tensor& input_r, const Tensor& weight_r, const Tensor& bias_r,
    IntList stride_, IntList padding_, IntList dilation_,
    bool transposed_, IntList output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled) {

...

  if (params.is_depthwise(input, weight)) {
      /* output.resize_(output_size(input, weight)); */

...
      output = at::thnn_conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation);
  } else if (params.use_cudnn(input)) {

...
      output = at::cudnn_convolution(
          input, weight, bias,
          params.padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
   
  } else if (params.use_miopen(input)) {

...
      output = at::miopen_convolution(
          input, weight, bias,
          params.padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);

  } else if (params.use_mkldnn(input)) {
#if AT_MKLDNN_ENABLED()
...

    output = at::mkldnn_convolution(input, weight, bias, params.padding, params.stride, params.dilation, params.groups);
#endif
  } else {
    if (params.groups == 1) {
      output = at::_convolution_nogroup(
          input, weight, bias, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
    } else {
...
        outputs[g] = at::_convolution_nogroup(
            input_g, weight_g, bias_g, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
      }
      output = at::cat(outputs, 1);
    }
  }

  if (k == 3) {
    output = view3d(output);
  }

  return output;
}    

```

```C++
// A generic function for convolution implementations which don't
// natively implement groups (e.g., not CuDNN).
at::Tensor _convolution_nogroup(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntList stride, IntList padding, IntList dilation,
    bool transposed, IntList output_padding) {

...

    if (dim == 4) {
      if (dilated) {
        return at::thnn_conv_dilated2d(
            input, weight, kernel_size, bias,
            stride, padding, dilation);
      } else {  /* dim == 4, non-dilated */
        /* CPU implementation has specialized MM kernels
           for non-dilated case here */
        return at::thnn_conv2d(
            input, weight, kernel_size, bias,
            stride, padding);
      }
    } else if (dim == 5 && (input.type().is_cuda() || dilated)) {
      return at::thnn_conv_dilated3d(
          input, weight, kernel_size, bias,
          stride, padding, dilation);
    } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
      /* CPU implementation has specialized MM kernels
         for non-dilated case here */
      return at::thnn_conv3d(
          input, weight, kernel_size, bias,
          stride, padding);
    }
..

  AT_ERROR("unsupported ConvNd parameters");
}

```

In case when cuDNN is supported, after few more hopes we reach the following:
```C++
void raw_cudnn_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getCudnnDataType(input);

...

  AT_CUDNN_CHECK(cudnnConvolutionForward(
    args.handle,
    &one, args.idesc.desc(), input.data_ptr(),
    args.wdesc.desc(), weight.data_ptr(),
    args.cdesc.desc(), fwdAlg, workspace.data, workspace.size,
    &zero, args.odesc.desc(), output.data_ptr()));
}
```

Here we are..
We finaly get to cuDNN SDK function [cudnnConvolutionForward()](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionForward).
"The cuDNN is closed-source low-level library for deep learning primitives developed by NVIDIA"
