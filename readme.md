#  Catcoon Pytorch Model

这个 repo 用于提供 [Catcoon](https://github.com/i-evi/catcoon) 的 `demo` 中给出的模型的 Pytorch 实现，用来训练模型并转换参数文件。

## 二进制参数流文件

在此特别对 Catcoon 的功能进行解释。在未制定参数文件规范时，可以使用二进制流文件加载参数:

```c
cc_tensor_t *cc_load_bin(const char *filename,
	const cc_int32 *shape, cc_dtype dtype, const char *name);
```

`cc_load_bin` 可以从一个二进制流文件中创建并加载一个 tensor，需要给出数据类型和 tensor 的形状:

```c
cc_int32 shape_conv1_w[] = {3, 3, 3, 32, 0};
conv1_w = cc_load_bin("conv1_w.bin", shape_conv1_w, CC_FLOAT32, "conv1_w");
```

上例展示了从`"conv1_w.bin"` 中加载一个形状为`[3, 3, 3, 32]` 的 tensor，数据类型是单精度浮点数(`CC_FLOAT32`)。 

由于目前没有尝试转换 Pytorch 的模型文件，参数的导出通过直接将模型参数 tensor 中的数据直接写入到流文件实现的。为了方便二进制文件的读写，需要使用 `ctypes` 通过 Native 方法读写二进制文件。

## 字节缓冲器(Bytes Buffer, bbuffer)

执行 `make` 编译 `bbuffer.c`，得到 `bbuffer.so`，即字节缓冲器。

字节缓冲器支持的数据类型在 `bbuffer.c` 中定义:

```c
#define float32 float
#define float64 double

WRITE_DT_IMPLEMENTATION(float32)
WRITE_DT_IMPLEMENTATION(float64)
```

在 Python 中通过 `ctypes` 调用:

```python
from ctypes import *
bbuffer = CDLL("bbuffer.so") # 注意，设置路径
bbuffer.write_float32.argtypes = (c_float,)
bbuffer.write_float32.restypes = c_float
...
```
在写二进制文件时，应该首先创建文件:

```python
filename = "conv1_w.bin"
bbuffer.init(filename.encode())
```

创建文件后可以向文件中写入二进制数据，例如把一个单精度浮点数 elem 写入流文件:

```python
elem = ...
bbuffer.write_float32(elem)
```
注意，流文件按照顺序进行写操作，不能够进行 `seek` 操作。此外，请留意主机的字节序(Endian)。

当一个文件的内容写入完成后，不要忘记关闭流文件缓冲器:

```python
bbuffer.close()
```
*`bbuffer` 不支持同时打开多个文件。

## 提供的模型示例

在运行所有示例前，不要忘记先编译 `bbuffer`，以便转换参数文件。

* LeNet，路径：`./lenet`，一个修改于 LeNet 的网络，与 LeNet 有共同的技术特征，包括卷积，池化，全连接，Relu、Softmax 激活函数*。

*这是一个 Helloworld 级的例子，此例的的文档交代了 Pytorch 模型的网络参数保存到二进制流文件的详细过程。这些内容在其他网络模型的文档中将不会重复出现。
