# HWTL_SDU-ANNS

# Introduction

#### Nearest neighbor search algorithm qsgngt, based on NGT-qg、Efanna、SSG, providing Python API.

# Install & Usage

## 1. **Installation**

### 1.1 Configure environment dependencies (recommended in the Docker container created by the Ubuntu image)

```bash
sudo apt update && sudo apt install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev

pip3 install wheel pybind11==2.5.0
```

### 1.2 Download code

```bash
git clone https://github.com/WPJiang/HWTL_SDU-ANNS.git
```

### 1.3 File copy & Set permissions

```bash
sudo cp HWTL_SDU-ANNS/lib/* /usr/local/lib/
sudo cp HWTL_SDU-ANNS/bin/* /usr/local/bin/
sudo chmod a+x /usr/local/bin/* && chmod a+x HWTL_SDU-ANNS/*
```

### 1.4 Configuration & installation

```bash
ldconfig
pip3 install HWTL_SDU-ANNS/qsgngt-*-linux_x86_64.whl
```

## 2. **Usage**

```python
python3 qsgngt.py
```

### The following describes the algorithm execution logic of qsgngt.py in steps

### 2.1 Data load

#### Method 1: Randomly generate data

```python
n, d = 10000, 128
X = np.random.randn(n, d)
```

#### Method 2: Load data from dataset (sift_base.fvecs as an example)

```python
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

X = fvecs_read("sift_base.fvecs")
```

### 2.2 Build Index (Divided into AKNNG, SG, QSG three stages)

#### 2.2.1 Build AKNNG

##### Method 1 (Using randomly generated data)

##### Execute ngt using subprocess or the command line, and then use ngtpy to incrementally build AKNNG

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

##### Method 1 (Using dataset data)

##### After reading the data set into the variable X, the rest of the operation is the same as randomly generating the data

##### Method 2 (Using randomly generated data)

##### Export X to fvecs format. For angular distance data sets, X needs to be normalized ahead of time

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

##### Build AKNNG using os.system or the command line (where fvecs is where the data set is stored, KNNG is where the KNNG is stored, and SG is where the SpaceGraph is stored)

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

##### Method 2 (Using dataset data)

##### For the angular distance data set, you need to import it for normalization processing, and then run the os.system or command line to invoke qsgngt-knng and qsgngt-SpaceGraph

##### Taking the SIFT1M dataset as an example, the sample parameters are as follows

##### Method 1：

| Dataset | metirc | dim | edge_size | epsilon | range | threshold | rangeMax | searchA | ifES |
| ------- | ------ | --- | --------- | ------- | ----- | --------- | -------- | ------- | ---- |
| SIFT1M  | 2      | 128 | 100       | 0.08    | 200   | 60        | 200      | 400     | 0    |

##### Method 2：

| Dataset | K   | L   | iter | S   | R   | SL  | SR  | SAngle |
| ------- | --- | --- | ---- | --- | --- | --- | --- | ------ |
| SIFT1M  | 400 | 400 | 12   | 10  | 100 | 100 | 100 | 60     |

#### 2.2.2 Build SG

##### Using subprocess or the command line

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

##### Taking the SIFT1M dataset as an example, the sample parameters are as follows

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

##### Taking the SIFT1M dataset as an example, the sample parameters are as follows

| Dataset | sample | max_edge_size |
| ------- | ------ | ------------- |
| SIFT1M  | 4000   | 96            |

### 2.3 Search

#### 2.3.1 Import the QSG graph index, the index address is `/qg/grp` in the storage index directory

```python
self.index = ngtpy.QuantizedIndex(index, self._max_edge_size)
self.index.set_with_distance(False)
self.indexName = index
```

#### 2.3.2 Set query parameters (Using a set of query parameters in the SIFT1M dataset as an example)

```python
self.index.set(epsilon=epsilon, result_expansion=result_expansion)
```

##### Taking the SIFT1M dataset as an example, the sample parameters are as follows（Including the corresponding QPS and Recall@10）

| Dataset | epsilon | result_expansion | QPS  | Recall@10 |
| ------- | ------- | ---------------- | ---- | --------- |
| SIFT1M  | 1.04    | 2.00             | 9902 | 0.9919    |

#### 2.3.3 Load query data

##### Method 1: Random generation

```python
n, d = 1000, 128
Y = np.random.randn(n, d)
```

##### Method 2: From Dataset (sift_query.fvecs as an example)

```python
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

Y = fvecs_read("sift_base.fvecs")
```

#### 2.3.4 Query

```python
for v in Y:
	self.index.search(v, n)
```

##### Where v represents the query vector and n represents the number of nearest neighbors returned

## 3. **Team members**
The team members come from HWTL and SDU. 
HWTL: Weipeng Jiang, Wei Han, Bo Bai
SDU: Kun Wang, Zixu Li, Zhiwei Chen, Shanzhi Li, Yupeng Hu, Liqiang Nie

