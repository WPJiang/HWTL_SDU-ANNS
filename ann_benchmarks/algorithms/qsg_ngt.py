from __future__ import absolute_import
import sys
import os
import ngtpy
import numpy as np
import subprocess
import time
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR

class QSG(BaseANN):
    # 参数初始化
    def __init__(self, metric, object_type, epsilon, param): 
        metrics = {'euclidean': '2', 'angular': 'E'} #metrics是一个字典，包含两个键值对，分别表示欧几里得距离和角距离
        self._edge_size = int(param['edge']) # 边数，即邻居数
        self._outdegree = int(param['outdegree']) # 出度
        self._indegree = int(param['indegree']) # 入度
        self._max_edge_size = int(param['max_edge']) if 'max_edge' in param.keys() else 128 # 最大边数，从参数列表中获取（若没有则默认为128）
        self._range = int(param['range']) # ANNG 构建的参数range，默认值为100
        self._metric = metrics[metric] # metric的取值为euclidea或angular；self._metric的取值为2或E，分别表示欧几里得距离、角距离
        self._object_type = object_type
        self._edge_size_for_search = int(param['search_edge']) if 'search_edge' in param.keys() else -2 # 搜索时的边数？邻居数？从参数列表中获取（若没有则默认为-2）
        self._tree_disabled = (param['tree'] == False) if 'tree' in param.keys() else False # 是否使用树索引，从参数列表中获取（若没有则默认为Fale）
        self._build_time_limit = 4 # 计时器限制为4
        self._epsilon = epsilon # 定义epsilon
        print('QSG: edge_size=' + str(self._edge_size))
        print('QSG: outdegree=' + str(self._outdegree))
        print('QSG: indegree=' + str(self._indegree))
        print('QSG: edge_size_for_search=' + str(self._edge_size_for_search))
        print('QSG: epsilon=' + str(self._epsilon))
        print('QSG: metric=' + metric)
        print('QSG: object_type=' + object_type)
        print('QSG: range=' + str(self._range))

    # 索引初始化
    def fit(self, X):
        print('QSG: start indexing...')
        dim = len(X[0]) # 数据维度dimension
        print('QSG: # of data=' + str(len(X))) # 数据条数（即对象个数）
        print('QSG: dimensionality=' + str(dim))
        index_dir = 'indexes' # 索引文件地址，该地址下会存储多个索引文件，如ONNG-{}-{}-{},ANNG-{}等
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        # index是onng索引的存放路径
        index = os.path.join(
            index_dir,
            'ONNG-{}-{}-{}'.format(self._edge_size, self._outdegree,
                                   self._indegree))
        # anngIndex是anng索引的存放路径
        anngIndex = os.path.join(index_dir, 'ANNG-' + str(self._edge_size))
        print('QSG: index=' + index)
        # 接下来使用多个if语句判断当前索引的创建情况，判断创建到了索引的第几步，并继续执行创建过程，直至完成索引创建
        if (not os.path.exists(index)) and (not os.path.exists(anngIndex)):# 如果该路径下既没有anng索引文件，也没有onng索引文件，那么就开始create ANNG
            print('QSG: create ANNG')
            t = time.time()
            args = ['ngt', 'create', '-it', '-p8', '-b500', '-ga', '-of',
                    '-D' + self._metric, '-d' + str(dim), # -D 的取值为2或E，分别表示欧几里得距离和角距离 ； -d 表示数据维度dimension
                    '-E' + str(self._edge_size), '-S40', # -E是edge_size
                    '-e' + str(self._epsilon), '-P0', '-B30', # -e是epsilon
                    '-T' + str(self._build_time_limit), '-R' + str(self._range), anngIndex] # -T 是self._build_time_limit ；-R 是 ANNG 构建的参数range；anngIndex是存放anng的地址
            subprocess.call(args) # subprocess.call()可以调用windows系统cmd命令行，根据args参数，执行额外的命令。等待，直至该命令完成或超时，然后得到返回码
            idx = ngtpy.Index(path=anngIndex) # 存入内存
            idx.batch_insert(X, num_threads=24, debug=False)
            idx.save()
            idx.close()
            print('QSG: ANNG construction time(sec)=' + str(time.time() - t))
        if not os.path.exists(index):# 如果不存在onng（前一个if语句已保证存在anng），那么就执行度调整degree adjustment
            print('QSG: degree adjustment')
            t = time.time()
            args = ['ngt', 'reconstruct-graph', '-mS',
                    '-E ' + str(self._outdegree), # 比onng-ngt.py中该命令多了-E这个参数
                    '-o ' + str(self._outdegree),
                    '-i ' + str(self._indegree), anngIndex, index]
            subprocess.call(args)
            print('QSG: degree adjustment time(sec)=' + str(time.time() - t))
        if not os.path.exists(index + '/QSG'):# 如果不存在qsg，那么就执行量化quantization
            print('QSG: quantization')
            t = time.time()
            args = ['ngtqsg', 'quantize', index]
            subprocess.call(args)
            print('QSG: quantization time(sec)=' + str(time.time() - t))
        if os.path.exists(index):# 如果存在经过了量化的onng（前面的if语句已保证该onng是经过了量化的onng），就表示图索引已经存在
            print('QSG: index already exists! ' + str(index))
            t = time.time()
            self.index = ngtpy.QuantizedIndex(index, self._max_edge_size)
            self.index.set_with_distance(False)
            self.indexName = index
            print('QSG: open time(sec)=' + str(time.time() - t))
        else:
            print('QSG: something wrong.')
        print('QSG: end of fit') # 索引初始化完成

    # 设置查询参数
    def set_query_arguments(self, parameters):
        result_expansion, epsilon = parameters
        print("QSG: result_expansion=" + str(result_expansion))
        print("QSG: epsilon=" + str(epsilon))
        self.name = 'QSG-NGT(%s, %s, %s, %s, %s, %1.3f)' % (
            self._edge_size, self._outdegree,
            self._indegree, self._max_edge_size,
            epsilon,
            result_expansion)
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, result_expansion=result_expansion)

    # 执行查询过程
    def query(self, v, n):
        return self.index.search(v, n)

    # 查询结束，释放索引
    def freeIndex(self):
        print('QSG: free')