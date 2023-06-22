import os
import struct
import subprocess
import time
import ngtpy
from sklearn import preprocessing
import numpy as np


def AKNNG(X, aknngIndex, _metric, dim, _edge_size, _epsilon, _range, _threshold, _rangeMax, _searchA, _ifES):
    args = [
        "ngt",
        "create",
        "-it",
        "-p8",
        "-b500",
        "-ga",
        "-of",
        "-D" + _metric,
        "-d" + str(dim),
        "-E" + str(_edge_size),
        "-S40",
        "-e" + str(_epsilon),
        "-P0",
        "-B30",
        "-T4",
        "-R" + str(_range),
        "-t" + str(_threshold),
        "-M" + str(_rangeMax),
        "-A" + str(_searchA),
        "-H" + str(_ifES),
        aknngIndex,
    ]
    subprocess.call(args)
    idx = ngtpy.Index(path=aknngIndex)
    idx.batch_insert(X, num_threads=24, debug=False)
    idx.save()
    idx.close()


def AKNNG_SG(aknngIndex, _metric, K, L, iter, S, R, SL, SR, SAngle, ifES):
    if ifES == 0:
        return
    if _metric == "E":
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
    graph_dir = 'graph'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    KNNG = os.path.join(graph_dir, 'KNNG-' + str(K) + '-' + str(L) + '-' + str(
        iter) + '-' + str(S) + '-' + str(R) + '.graph')
    SG = os.path.join(aknngIndex, 'grp')
    cmds = (
            "/home/app/hwtl_sdu-anns-qsgngtlib/qsgngt-knng "
            + str(fvecs)
            + " "
            + str(KNNG)
            + " "
            + str(K)
            + " "
            + str(K)
            + " "
            + str(iter)
            + " "
            + str(S)
            + " "
            + str(R)
            + "&& /home/app/hwtl_sdu-anns-qsgngtlib/qsgngt-SpaceGraph "
            + str(fvecs)
            + " "
            + str(KNNG)
            + " "
            + str(SL)
            + " "
            + str(SR)
            + " "
            + str(SAngle)
            + " "
            + str(SG)
    )
    os.system(cmds)


def SG(aknngIndex, index, _outdegree, _indegree):
    print("QSG: SG")
    t = time.time()
    args = [
        "ngt",
        "reconstruct-graph",
        "-mS",
        "-E " + str(_outdegree),
        "-o " + str(_outdegree),
        "-i " + str(_indegree),
        aknngIndex,
        index,
    ]
    subprocess.call(args)
    print("QSG: SG construction time(sec)=" + str(time.time() - t))


def QSG(index, _sample, _max_edge_size):
    print("QSG:create and append...")
    t = time.time()
    args = ["qbg", "create-qg", index]
    subprocess.call(args)
    print("QSG: create qsg time(sec)=" + str(time.time() - t))
    print("QB: build...")
    t = time.time()
    args = [
        "qbg",
        "build-qg",
        "-o" + str(_sample),
        "-M6",
        "-ib",
        "-I400",
        "-Gz",
        "-Pn",
        "-E" + str(_max_edge_size),
        index,
    ]
    subprocess.call(args)
    print("QSG: build qsg time(sec)=" + str(time.time() - t))


def search(index, _max_edge_size):
    if os.path.exists(index + "/qg/grp"):
        print("QSG: index already exists! " + str(index))
        t = time.time()
        qsg_index = ngtpy.QuantizedIndex(index, _max_edge_size)
        qsg_index.set_with_distance(False)
        indexName = index
        print("QSG: open time(sec)=" + str(time.time() - t))
        # 搜索
        for v in Y:
            print(qsg_index.search(v, n))
    else:
        print("QSG: something wrong.")


if __name__ == "__main__":
    n, d = 10000, 128
    X = np.random.randn(n, d)
    n, d = 1000, 128
    Y = np.random.randn(n, d)
    _metric = '2'
    dim = d
    _edge_size = 100
    _epsilon = 0.08
    _range = 200
    _threshold = 60
    _rangeMax = 200
    _searchA = 400
    _ifES = 1  # 若为1，则使用efanna、nssg的路线创建aknng
    _outdegree = 64
    _indegree = 120
    _sample = 4000
    _max_edge_size = 96
    index_dir = "indexes"
    print("QSG: index")
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    index = os.path.join(index_dir, "SG-{}-{}-{}".format(_edge_size, _outdegree, _indegree))
    aknngIndex = os.path.join(index_dir, "AKNNG-" + str(_edge_size))
    # 构建AKNNG
    AKNNG(X, aknngIndex, _metric, dim, _edge_size, _epsilon, _range, _threshold, _rangeMax, _searchA, _ifES)
    AKNNG_SG(aknngIndex, _metric, 100, 100, 10, 8, 10, 100, 100, 60, _ifES)
    # 构建SG
    SG(aknngIndex, index, _outdegree, _indegree)
    # 构建QSG
    QSG(index, _sample, _max_edge_size)
    # 引入索引并搜索
    search(index, _max_edge_size)
