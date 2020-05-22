# LeNet

这是一个简单的例子，相对原版的 LeNet 稍作修改，旨在识别手写数字，运行第一个使用 Catcoon 的 CNN 网络吧。

## 网络模型和训练(`train.py`)

网络模型直接定义在 `train.py` 中:

```python
self.conv = nn.Sequential(
    nn.Conv2d(1, 32, 5, 1, 2, bias=True),
    nn.ReLU(),
	nn.MaxPool2d(2, 2),
	nn.Conv2d(32, 64, 5, 1, 2, bias=True),
	nn.ReLU(),
	nn.MaxPool2d(2, 2),
)
self.fc = nn.Sequential(
	nn.Linear(7 * 7 * 64, 128),
	nn.ReLU(),
	nn.Linear(128, 10),
	nn.Softmax()
```

训练模型，会自动下载数据集(如果数据集不存在)，并且保存模型文件:
```bash
$python train.py
```
训练完成后，将会保存一个模型文件 `classifier.pkl`。

## 转换网络参数到二进制流文件(`cvrt.py`)

运行下面的命令可以加载 `classifier.pkl` 并把参数以二进制流的形式保存到文件中。

```bash
$python cvrt.py
```
Pytorch 的 `nn.Module` 对象可以通过 `state_dict()` 方法得到参数 `dict`:

```python
class classifier(nn.Module):
	pass
C = torch.load("classifier.pkl") # 加载模型，不要忘记先训练好模型
state_dict = C.state_dict()      # 获取参数 dict
```

使用 `torch.nn.Sequential` 创建的网络模型的参数会被自动命名(作为 `state_dict` 的索引)，其命名命名规则是按照网络结构的顺序进行的。某些结构并不具有可训练的参数，例如 `nn.ReLU`，`nn.MaxPool2d`，但是他们也被算作“一层”，直接用 `state_dict` 中的 tensor 命名文件显得不是那么符合我们的预期，例如第一个卷积层的参数命名为 `conv.0.weight` 和 `conv.0.bias`，而第二个卷积层的参数却被命名为 `conv.3.weight` 和 `conv.3.bias`，跳过的层编号 `1`，`2` 实际上被 `nn.ReLU` 和 `nn.MaxPool2d` 占用。但是好在这些参数 tensor 在 `dict` 中总是按网络结构顺序排列的，因此，我们可以根据网络结构，知道我们想要的参数都有哪些，并给打算用来保存他们的文件按顺序命名:

```python
parameters_files = [
	"conv1_w.bin",
	"conv1_b.bin",
	"conv2_w.bin",
	"conv2_b.bin",
	"fc1_w.bin",
	"fc1_b.bin",
	"fc2_w.bin",
	"fc2_b.bin",
]
```

如果觉得网络参数过多，指定一个目录统一存放这些文件:

```python
parameters_filepath = "./dump"
```

遍历 `state_dict` 用 tensor 名字作为索引逐个保存参数 tensor 到二进制流文件:

```python
for i, name in enumerate(state_dict):
	dump_parameters(state_dict, name,
		parameters_filepath, parameters_files[i])
```
`dump_parameters` 的实现如下:

```python
def dump_parameters(state_dict, name, filepath, filename):
	current_file = "%s/%s"%(filepath, filename)
	bbuffer.init(current_file.encode())
	para = state_dict[name].numpy().reshape(-1)
	for elem in para:
		bbuffer.write_float32(elem)
	bbuffer.close()s
```

很简单的逻辑，初始化 `bbuffer`，打开文件，写入全部参数后关闭文件。

执行了 `cvrt.py` 后，转换过的参数二进制流文件保存在 `./dump` 路径下:

```bash
$ls dump/
conv1_b.bin  conv2_b.bin  fc1_b.bin  fc2_b.bin
conv1_w.bin  conv2_w.bin  fc1_w.bin  fc2_w.bin
```

## 编译运行 catcoon 的 LeNet(`lenet.c`)

首先获得 [Catcoon](https://github.com/i-evi/catcoon)，`lenet.c` 位于 catcoon repo 的 `./demo` 路径下。在 [Catcoon](https://github.com/i-evi/catcoon) repo 的根目录下的 `makefile` 中，`lenet` 已经在 `APP_NAMES` 中，因此，直接 `make` 就会编译 `lenet.c`。

编译前，应该先编辑 `lenet.c`，配置参数文件的路径和参数文件名:

```c
const char *parameters_files[]={
	"conv1_w.bin",
	"conv1_b.bin",
	"conv2_w.bin",
	"conv2_b.bin",
	"fc1_w.bin",
	"fc1_b.bin",
	"fc2_w.bin",
	"fc2_b.bin",
};
const char *parameters_path = "/home/evi/catcoon-pytorch-model/lenet/dump";
```
`parameters_path` 可以用相对路径，也可以是绝对路径，你也可以修改代码使得路径可以通过 `main` 函数的参数给出。请确保已经训练并转换过参数。

在 [Catcoon](https://github.com/i-evi/catcoon) repo 下执行 `make` 即可编译。编译完成后，找一张图像试试吧！

```bash
$./lenet /home/evi/catcoon-pytorch-model/imgs/mnist9.bmp
```
如果不出意外，你可能会看到这样的输出:

```bash
[0]: 0.000000
[1]: 0.000000
[2]: 0.000000
[3]: 0.000004
[4]: 0.000608
[5]: 0.000004
[6]: 0.000000
[7]: 0.000087
[8]: 0.000008
[9]: 0.999288
Result of "/home/evi/catcoon-pytorch-model/imgs/mnist9.bmp": [9]
```