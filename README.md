
## 如何在colab上运行代码?

### 方式1: 读取google driver上的数据速度慢

将code上传到google driver上, 来执行, 假如google driver上的目录结构如下:

```
MyDrive/
       /data/      - 保存数据
       /relaxdl/   - 保存代码
!python3 faster_rcnn/util.py
```

* 默认的工作目录是: ```/content```
* 我们需要设置工作目录为: ```/content/drive/MyDrive/relaxdl```, 这样做是因为我们数据默认的缓存路径是```../data```, 在google driver上而不是虚拟机上, 切换虚拟机后, 缓存的数据仍然在, 不需要重新download
* 这种方式最大的一个缺点就是从google driver读取数据的速度很慢

```python
import os
from google.colab import drive

drive.mount('/content/drive', force_remount=True)
working_path = "/content/drive/MyDrive/relaxdl"
os.chdir(working_path)

# 执行测试代码
!python3 pytorch_basic/faster_rcnn/train.py
```

### 方式2: 数据拷贝到虚拟机磁盘再运行

* 我们要把工作目录切换回: ```/content```
* 下面的代码会把数据从google driver上拷贝到虚拟机本地的目录: ```/data```
* 缺点是切换虚拟机之后, 数据需要重新拷贝一边

```python
import shutil
os.chdir('/content')

# 拷贝文件夹:
shutil.copytree('/content/drive/MyDrive/data/VOCdevkit', '/data/VOCdevkit')
# 拷贝单个文件:
shutil.copy('/content/drive/MyDrive/data/VOCdevkit.zip', '/data')

# 执行测试代码
!python3 /content/drive/MyDrive/relaxdl/pytorch_basic/faster_rcnn/train.py
```

### 导入本地模块执行

需要手动添加模块搜索路径

```python
import sys
sys.path.append('/content/drive/MyDrive/relaxdl/pytorch_basic/faster_rcnn/')

import train
train.run()
```

