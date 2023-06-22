# HWTL_SDU-ANNS

# Introduction

#### Nearest neighbor retrieval algorithm qsgngt, based on NGT-qg、Efanna、SSG, providing Python API

# Install & Usage

## 1. **Installation**

### 1.1 配置环境依赖(建议在 Ubuntu 镜像创建的 Docker 容器中执行)

```bash
sudo apt update && sudo apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev

pip3 install wheel pybind11==2.5.0
```

### 1.2 下载代码

```bash
git clone https://github.com/WPJiang/HWTL_SDU-ANNS.git
```

### 1.3 文件复制及赋予权限

```bash
sudo cp HWTL_SDU-ANNS/lib/* /usr/local/lib/
sudo cp HWTL_SDU-ANNS/bin/* /usr/local/bin/
sudo chmod a+x /usr/local/bin/* && chmod a+x HWTL_SDU-ANNS/*
```

### 1.4 配置安装

```bash
ldconfig
pip3 install HWTL_SDU-ANNS/qsgngt-*-linux_x86_64.whl
```

## 2. **Usage**

```python
python3 qsgngt.py
```

### 下面分步骤介绍 qsgngt.py 的算法执行逻辑

### 2.1 数据处理/获取

#### 方法 1：随机生成数据

```python
n, d = 10000, 128
X = np.random.randn(n, d)
```

#### 方法 2：使用数据集载入数据（以 sift_base.fvecs 为例）

```python
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

X = fvecs_read("sift_base.fvecs")
```

### 2.2 Build Index（分为 AKNNG，SG，QSG 三个阶段）

#### 2.2.1 Build AKNNG

##### 方法 1（以随机生成数据为例）

##### 使用 subprocess 或者命令行执行 ngt, 再使用 ngtpy 来对 AKNNG 进行增量构建

```python
import subprocess
import ngtpy

args = [
                "ngt",
                "create",
                "-it",
                "-p8",
                "-b500",
                "-ga",
                "-of",
                "-D" + self._metric,
                "-d" + str(dim),
                "-E" + str(self._edge_size),
                "-S40",
                "-e" + str(self._epsilon),
                "-P0",
                "-B30",
                "-T4",
                "-R" + str(self._range),
                "-t" + str(self._threshold),
                "-M" + str(self._rangeMax),
                "-A" + str(self._searchA),
                "-H" + str(self._ifES),
                anngIndex,
            ]
	 subprocess.call(args)
   idx = ngtpy.Index(path=anngIndex)
   idx.batch_insert(X, num_threads=24, debug=False)
   idx.save()
   idx.close()
```

##### 方法 1（以数据集数据为例）

##### 将数据集读入后记为变量 X，其余操作与随机生成数据相同

##### 方法 2（以随机生成数据为例）

##### 将 X 导出为 fvecs 格式。如果为角距离数据集,则需要将 X 提前进行归一化操作

```python
from sklearn import preprocessing
if self._metric == "E":
    X_normalized = preprocessing.normalize(X, norm="l2")
    fvecs_dir = "fvecs"
    if not os.path.exists(fvecs_dir):
        os.makedirs(fvecs_dir)
    fvecs = os.path.join(fvecs_dir, "base.fvecs")
    with open(fvecs, "wb") as fp:
        for y in X_normalized:
            d = struct.pack("I", y.size)
            fp.write(d)
            for x in y:
                a = struct.pack("f", x)
                fp.write(a)
else:
    fvecs_dir = "fvecs"
    if not os.path.exists(fvecs_dir):
        os.makedirs(fvecs_dir)
    fvecs = os.path.join(fvecs_dir, "base.fvecs")
    with open(fvecs, "wb") as fp:
        for y in X:
            d = struct.pack("I", y.size)
            fp.write(d)
            for x in y:
                a = struct.pack("f", x)
                fp.write(a)
```

##### 使用 os.system 或者命令行构建 AKNNG（其中 fvecs 为数据集的存储位置, KNNG 为保存 KNNG 的位置, SG 为保存 SpaceGraph 的位置）

```python
cmds = (
                "/home/app/HWTL_SDU-ANNS/qsgngt-knng "
                + str(fvecs)
                + " "
                + str(KNNG)
                + " "
                + K
                + " "
                + L
                + " "
                + iter
                + " "
                + S
                + " "
                + R
                + "&& /home/app/HWTL_SDU-ANNS/qsgngt-SpaceGraph "
                + str(fvecs)
                + " "
                + str(KNNG)
                + " "
                + SL
                + " "
                + SR
                + " "
                + SAngle
                + " "
                + str(SG)
            )
            os.system(cmds)
```

##### 方法 2（以数据集数据为例）

##### 如为角距离数据集，则需要导入进行归一化处理后，再使用 os.system 或命令行调用 qsgngt-knng 和 qsgngt-SpaceGraph

##### 以 SIFT1M 数据集为例，示例参数如下

##### 方法 1：

| Dataset | metirc | dim | edge_size | epsilon | range | threshold | rangeMax | searchA | ifES |
| ------- | ------ | --- | --------- | ------- | ----- | --------- | -------- | ------- | ---- |
| SIFT1M  | 2      | 128 | 100       | 0.08    | 200   | 60        | 200      | 400     | 0    |

##### 方法 2：

| Dataset | K   | L   | iter | S   | R   | SL  | SR  | SAngle |
| ------- | --- | --- | ---- | --- | --- | --- | --- | ------ |
| SIFT1M  | 400 | 400 | 12   | 10  | 100 | 100 | 100 | 60     |

#### 2.2.2 Build SG

##### 使用 subprocess 或者命令行进行调用

```python
args = [
		"ngt",
		"reconstruct-graph",
		"-mS",
		"-E " + str(outdegree),
		"-o " + str(outdegree),
		"-i " + str(indegree),
		anngIndex,
		index,
]
subprocess.call(args)
```

##### 以 SIFT1M 数据集为例，示例参数如下

| Dataset | outdegree | indegree |
| ------- | --------- | -------- |
| SIFT1M  | 64        | 120      |

#### 2.2.3 Build QSG

```python
args = ["qbg", "create-qg", index]
subprocess.call(args)
args = [
			"qbg",
			"build-qg",
			"-o" + str(sample),
			"-M6",
			"-ib",
			"-I400",
			"-Gz",
			"-Pn",
			"-E" + str(max_edge_size),
			index,
]
subprocess.call(args)
```

##### 以 SIFT1M 数据集为例，示例参数如下

| Dataset | sample | max_edge_size |
| ------- | ------ | ------------- |
| SIFT1M  | 4000   | 96            |

### 2.3 Search

#### 2.3.1 导入 QSG 图索引，索引地址为存储索引目录中的`/qg/grp`

```python
self.index = ngtpy.QuantizedIndex(index, self._max_edge_size)
self.index.set_with_distance(False)
self.indexName = index
```

#### 2.3.2 设置查询参数（以 SIFT1M 数据集中的一组查询参数为例）

```python
self.index.set(epsilon=epsilon, result_expansion=result_expansion)
```

##### 以 SIFT1M 数据集为例，示例参数如下，其对应的 QPS 和 Recall@10 如表格所示

| Dataset | epsilon | result_expansion | QPS  | Recall@10 |
| ------- | ------- | ---------------- | ---- | --------- |
| SIFT1M  | 1.04    | 2.00             | 9902 | 0.9919    |

#### 2.3.3 查询数据生成

##### 方法 1：随机生成

```python
n, d = 1000, 128
Y = np.random.randn(n, d)
```

##### 方法 2：数据集导入数据（以 sift_query.fvecs 为例）

```python
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

Y = fvecs_read("sift_base.fvecs")
```

#### 2.3.4 执行查询

```python
for v in Y:
	self.index.search(v, n)
```

##### 其中，v 代表 query 向量, n 代表返回的最近邻数目
