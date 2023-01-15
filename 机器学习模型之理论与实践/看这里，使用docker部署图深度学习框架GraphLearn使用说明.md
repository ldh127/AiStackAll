看这里，使用docker部署图深度学习框架GraphLearn使用说明

---

最近几年，**图深度学习(Graph DNN)** 火的如火如荼，图以其强大的 **关系建模** 能力和 **可解释** 能力，逐步在Embedding 算法设计技术中展露头角。在以前的两篇文章 [graphSage还是HAN ？吐血力作综述Graph Embeding 经典好文](https://mp.weixin.qq.com/s/T1oLgGMUEZYfTGrHVT7-9A) 和  [一文揭开图机器学习的面纱，你确定不来看看吗 ?](https://mp.weixin.qq.com/s/6dDz9QFjQOh5RuK4pMx9Kg) 中，作者分别对图的基础知识和 Graph Embeding 进行了讲解，让我们对图的基础概念有了大致的了解。其中，文章末尾有推荐到使用训练图深度的学习模型算法需要借助在 **深度学习框架** 之上开发的 **图深度学习框架** 这项技术。

大家都知道 **tensorflow** 或则 **pytorch**  是现在非常流行的两种 **DNN深度学习框架**，而实现 Graph Embeding 算法则需要使用 **图深度学习/机器学习框架**。现实学习工作中，很多人对tensorflow 的使用有着非常大的偏爱，这里作者也是不能免俗气，非常乐意使用基于tensorflow相关的产品。在作者开始了解基于图深度学习框架的时候，就发现了Graph learn 这个产品。

**GraphLearn** 以前也叫 **AliGraph**, 是阿里巴巴于2020年左右开源的一款简化 图神经网络(Graphic Nuaral Network, GNN) 应用的新框架。该框架**底层兼容 tensorflow 和pytorch**， 能够  **基于docker进行环境搭建**，非常方便上手。

但是作者在实际实践的过程中，却实实在在的踩了一堆坑，很多地方按照官方的文档确实没办法对框架进行使用，所以就自己摸索了一番，整理了一下搜集到的资料和框架使用相关的知识分享给大家～ 

---

### (1) GraphLearn 的docker过程 

注意：graph learn 框架部署成功的一款配置参数是：

       python===2.7
       tensorflow==1.13.1
       
采用 `pip2 install xxx` 进行python软件包的安装 。

---

#### (1.1)  常用文档 

在这里贴上 graph learn 相关的一些文档链接：

[官方文档](https://graph-learn.readthedocs.io/zh_CN/latest/zh_CN/gl/install.html)  ： https://graph-learn.readthedocs.io/zh_CN/latest/zh_CN/gl/install.html

[Github路径](https://github.com/alibaba/graph-learn/blob/master/README_cn.md)： https://github.com/alibaba/graph-learn/blob/master/README_cn.md

[阿里云开发者社区文档](https://developer.aliyun.com/article/752630)： https://developer.aliyun.com/article/752630

从上面的文档中，我们可以了解到更多graph learn 使用相关的底层使用相关的东西。

---

#### (1.2)  docker 安装使用Graph Learn 

当然，使用graph learn也可以自己去安装配置，但是中途坑太多并且很多 **错误太隐晦** ，不知道去哪里查找。所以这里推荐使用docker进行安装使用。

关于 docker 工具本身的安装可以参考作者以前的一篇文章： [docker 快速实战指南](https://mp.weixin.qq.com/s/phk68jKVP5HK0QcP2kr44g)  ，里面详细介绍了dock使用的基础知识。

---

##### (1.2.1)  拉取 graph learn镜像

我们可以使用以下命令去docker hub中拉取阿里巴巴官方打包好的镜像：

`sudo docker pull graphlearn/graphlearn:1.0.0-tensorflow1.13.0rc1-cpu
`

这里以基于tensorflow 的 cpu 版本为栗。
要使用gpu的版本的，也可以拉取 `1.0.0-torch1.8.1-cuda10.2-cudnn7` 相关版本，从名字我们可以看到它仅仅支持torch。暂时作者也没有找到支持tensorflow 的GPU版本，其中 **[docker hub](https://hub.docker.com)** 的链接如下：`https://hub.docker.com/` ， 搜索关键子是：**graphlearn/graphlearn** 。

执行完这个命令之后，我们就在本机器上可以看到当前镜像了。如下图所示：
![](https://files.mdnice.com/user/17436/ea450520-c290-4225-b168-efbed7c15f9c.jpg)

---

##### (1.2.2) 启动 graph learn 实例

上一步我们已经拉取好镜像了,下面我们可以使用下面的命令启动它：

`sudo docker run  -d -it  --name graph_m -v  /home/local_machine/workspace:/root/workspace  graphlearn/graph-learn`

**注意**： 这里我们可以把我们要使用的代码和数据放入到当前的目录中，通过虚拟卷的映射绑定映射到docker内部。

在后文，我会把自己使用代码和数据上传，读者可以把放到自己本地机器的 `/home/local_machine/workspace` 路径下。 
如下图所示：
![](https://files.mdnice.com/user/17436/ed05fa41-a376-498a-841b-b0da99041f2b.jpg)

---

##### (1.2.3) 进入graph leanr 实例
因为我们需要运用docker 装好的环境，所以我们是需要登录到docker内部去执行命令训练模型的。
这里我们可以采这个命令进入 docker实例 ：

`sudo docker exec -it 6c1c2dda75f9 /bin/bash`

如下图所示：
![](https://files.mdnice.com/user/17436/cbc3a9c5-0ed7-4238-bb44-fb77ba56f89b.jpg)

---

##### (1.2.4) 开始使用 graph learn 框架训练图机器学习模型

在这里，我们把训练模型需要的数据下载到 自己机器上的 `/home/local_machine/workspace`  这个路径下(可以改为你设置的路径，上面第二条-v 冒号前面的那个路径也要同步改掉)。

然后进入到镜像里面，开始训练模型了，输入数据和执行的代码选择  `/root/workspace` 这个路径。
然后执行：

`python train_tg_unsupervised.py `

这样，就可以**开始我们的模型训练之旅**了。

训练效果图如下：
![](https://files.mdnice.com/user/17436/25140fbe-0856-4fdc-9e38-efb7800485e1.jpeg)


注意： 我们这里是使用镜像的graph learn 环境，使用我们的数据，训练自己的模型。

当然，很多同学大概率会遇到这个问题：

![](https://files.mdnice.com/user/17436/b60fa06f-3e12-493f-9573-bf5bedf6773b.jpeg)


如果出现这个问题，不用怀疑，**大概率是数据(节点或则边)的格式不对**，并且**大概率是因为中间的TAB空格不对**。这里必须是 Linux下的TAB键，如果同学你用多个空格键代替TAB键，那恭喜同学，你会收到意想不到的惊喜，不说了，这些教训都是血与泪！！！

---

### (2) 代码时光
 
 
 ***talk is cheap , show me the code !!!*** 

下面的代码是使用tensorflow来构建模型结构，中间支持了 **metapath节点采样**，支持图上的 **node embeding导出**。 在下面提供了完整的代码和数据下载的方式，代码的注解非常详细，具有极高的参考价值与可重用性，有问题欢迎讨论～

该工程完整代码和训练数据，可以去算法全栈之路公众号回复 “**阿里图框架源码**” 下载。


```

@ 欢迎关注微信公众号：算法全栈之路
# -*- coding: utf-8 -*-


from __future__ import print_function
import graphlearn as gl
import tensorflow as tf
import graphlearn.python.nn.tf as tfg
from ego_sage import EgoGraphSAGE


def load_graph(config):
    # 数据涵盖顶点数据和边数据,顶点数据和边均可以有属性,并且顶点种类和边种类可以有多个,只是type字段不能相同
    # 读取图参数,构建一个 Graph 逻辑对象,后续所有的操作都在这个 Graph 对象上进行。
    # GL图数据格式灵活，支持float，int，string类型的属性，支持带权重、标签。
    # 数据源载入和拓扑描述：同构、异构、多数据源，通通支持
    # GL提供了 node 和 edge 两个简单的接口来支持顶点和边的数据源载入，同时在这两个接口中描述图的拓扑结构，比如“buy”边的源顶点类型是“user”，
    # 目的顶点类型是“item”。这个拓扑结构在异构图中十分重要，是后续采样路径meta-path的依据，也是查询的实体类型的基础。
    data_dir = config['dataset_folder']
    g = gl.Graph() \
        .node(data_dir + 'unsupervised_node.txt', node_type='i',
              decoder=gl.Decoder(attr_types=['float'] * config['features_num'],
                                 attr_dims=[0] * config['features_num'])) \
        .edge(data_dir + 'unsupervised_edge.txt', edge_type=('i', 'i', 'train'),
              decoder=gl.Decoder(weighted=True), directed=False)
    return g


def meta_path_sample(ego, ego_name, nbrs_num, sampler):
    # 构建 meta path 采样器
    """ creates the meta-math sampler of the input ego.
    config:
      ego: A query object, the input centric nodes/edges
      ego_name: A string, the name of `ego`.
      # 邻居个数,每一跳 选择几个邻居
      nbrs_num: A list, the number of neighbors for each hop.

      sampler: A string, the strategy of neighbor sampling.
    """
    # src_hop_i
    # 下游访问的时候根据alist 去访问
    alias_list = [ego_name + '_hop_' + str(i + 1) for i in range(len(nbrs_num))]
    for nbr_count, alias in zip(nbrs_num, alias_list):
        # 这里将ego采样后赋值给ego,完成二跳的采样
        # 在下游用alias进行数据访问
        ego = ego.outV('train').sample(nbr_count).by(sampler).alias(alias)
    return ego


def query(graph, config):
    # 拿到所有 label 叫做 train 的边,然后进行 batch 的shuffle
    seed = graph.E('train').batch(config['batch_size']).shuffle(traverse=True)

    # 得到 batch 之后的边, outv 是出顶点, src 是 out 对应的顶点
    # 当前batch 所有的边里所有的出另一个顶点,因为是有向图,所以有区分出度入度
    src = seed.outV().alias('src')

    # inv ,整个图是入度图, dst 是 in 对应的顶点
    # 当前batch里顶点的 ,所有的入顶点
    dst = seed.inV().alias('dst')

    # 使用 alias 可以在下游访问当前次查询输出的结果
    # 每种采样操作都有不同的实现策略，例如随机、边权等
    neg_dst = src.outNeg('train').sample(config['neg_num']).by(config['neg_sampler']).alias('neg_dst')

    # 异构图, meta_path 采样, 这个 [10,5 ] 作为
    # 输入 当前批次所有的 src,然后进行采样
    src_ego = meta_path_sample(src, 'src', config['nbrs_num'], config['sampler'])
    # 输入当前批次所有dst ,进行采样
    dst_ego = meta_path_sample(dst, 'dst', config['nbrs_num'], config['sampler'])

    # GL 也提供了负采样，只需要将示例中的 outV 改为 outNeg 即可
    # 输出 outNeg ,采样 neg_num 条, in_degree , 根据入度进行采样
    # 负采样，基于输入顶点，对不直接连接的顶点进行采样。负采样经常被用作监督学习的重要工具。
    dst_neg_ego = meta_path_sample(neg_dst, 'neg_dst', config['nbrs_num'], config['sampler'])

    # 一批一批的边的选择值,values是sink操作,表示查询已经结束
    return seed.values()


def train(graph, model, config):
    # config train 训练 模式,超参
    tfg.conf.training = True

    # 根据 config 从图上按批次获得数据,并得到对应的一些采样数据
    query_train = query(graph, config)

    # 窗口,当前batch ,因为是根据边采样,所以下游的src和dst 的embeding 应该是一一对应
    # graphlearn.Dataset接口，用于将 Query 的结果构造为 Numpy 组成的graphlearn.Nodes / graphlearn.Edges或graphlearn
    # window. The data set will be prefetched asynchronously, window is the size of the prefetched data.
    # 完成 numpy 到 tensor 的转换
    dataset = tfg.Dataset(query_train, window=10)

    # 得到 起始边 和 结束边
    # Origanizes the data dict as EgoGraphs and then check and return the specified `EgoGraph`.
    # 将包含固定大小样本的 GSL 结果转换为 EgoGraph 格式。您可以 EgoGraph 通过别名获得 ego graph
    src_ego = dataset.get_egograph('src')
    dst_ego = dataset.get_egograph('dst')
    neg_dst_ego = dataset.get_egograph('neg_dst')

    # 在图上跑模型,得到 三个 logit
    # 这里模型的forword其实就是结点特征的聚合,聚合完了之后就计算得到logit
    src_emb = model.forward(src_ego)
    dst_emb = model.forward(dst_ego)
    neg_dst_emb = model.forward(neg_dst_ego)
    # use sampled softmax loss with temperature.
    # 温度值,应该是相似度到sigmoid 值之间的那个转化方式
    # 无监督类型网络,有边相连的,就比较近, 没有边相连的,随机采样,距离应该比较远
    loss = tfg.unsupervised_softmax_cross_entropy_loss(src_emb, dst_emb, neg_dst_emb,
                                                       temperature=config['temperature'])
    return dataset.iterator, loss


# node_embedding(g, u_model, 'u', config)
def node_embedding(graph, model, node_type, config):
    """ save node embedding.
    Args:
      node_type: 'u' or 'i'.
    Return:
      iterator, ids, embedding.
    """
    tfg.conf.training = False
    ego_name = 'save_node_' + node_type
    seed = graph.V(node_type).batch(config['batch_size']).alias(ego_name)
    nbrs_num = config['nbrs_num'] if node_type == 'u' else config['nbrs_num']

    query_save = meta_path_sample(seed, ego_name, config['nbrs_num'], config['sampler']).values()
    dataset = tfg.Dataset(query_save, window=1)
    ego_graph = dataset.get_egograph(ego_name)
    emb = model.forward(ego_graph)
    return dataset.iterator, ego_graph.src.ids, emb


def dump_embedding(sess, iter, ids, emb, emb_writer):
    sess.run(iter.initializer)
    while True:
        try:
            outs = sess.run([ids, emb])
            # [B,], [B,dim]
            feat = [','.join(str(x) for x in arr) for arr in outs[1]]
            for id, feat in zip(outs[0], feat):
                emb_writer.write('%d\t%s\n' % (id, feat))
        except tf.errors.OutOfRangeError:
            print('Save node embeddings done.')
            break


def run(config):
    # graph input data
    # load 已经定义好的图
    g = load_graph(config=config)
    # GL提供单机的版本， 通过init 接口快速启动Graph Engine，至此，图对象已经构造完毕
    # 查询、采样操作就可以在 Graph 上进行了。
    g.init()

    # Define Model
    # 根据读入的参数进行模型各层的参数,初始第一层是每个结点的特征个数或则embeding 维度。
    # 后面层是对邻居结点做卷积或则说是每2跳之间结点特征聚合的过程,每次聚合需要用到2跳结点的特征.
    # dims = [128, 128, 10, 5, 128 ]
    dims = [config['features_num']] + [config['hidden_dim']] * (len(config['nbrs_num']) - 1) + [config['output_dim']]
    # graphsage,隐藏层参数的多少
    # 定义一个模型,该模型内部,进行了featurecolumn 的 embeding化
    # 这个维度第一层是features_nums,也就是结点的初始特征值,初始特征可能需要使用featurecolumn 经过transform,
    # 初始特征是经过多层的聚合,维度会逐渐变小,图上的每个结点是需要初始的embeding的,或则很多维数特征
    # 每一个 embeding ,经过几层神经网络之后,维度变成 1dim,计算出 logit,这里看 ego_gnn里的源码
    # graphsage,图迭代几次,就使用几跳的结点数据进行sage操作
    model = EgoGraphSAGE(dims,
                         agg_type=config['agg_type'],
                         dropout=config['drop_out'])

    # train
    iterator, loss = train(g, model, config)

    optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
    train_op = optimizer.minimize(loss)
    train_ops = [loss, train_op]

    # 保存下来embeding
    i_save_iter, i_ids, i_emb = node_embedding(g, model, 'i', config)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        step = 0
        print("Start Training...")
        for i in range(config['epoch']):
            try:
                while True:
                    ret = sess.run(train_ops)
                    print("Epoch {}, Iter {}, Loss {:.5f}".format(i, step, ret[0]))
                    step += 1
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)  # reinitialize dataset.

        print("Start saving embeddings...")
        i_emb_writer = open('i_emb.txt', 'w')
        i_emb_writer.write('id:int64\temb:string\n')
        dump_embedding(sess, i_save_iter, i_ids, i_emb, i_emb_writer)

    g.close()


if __name__ == "__main__":
    # 模型训练超参
    config = {'dataset_folder': './data/tg_graph/',
              'batch_size': 2,
              'features_num': 3,
              'hidden_dim': 3,
              'output_dim': 3,
              'nbrs_num': [1],
              'neg_num': 1,
              'learning_rate': 0.0001,
              'epoch': 1,
              'agg_type': 'mean',
              'drop_out': 0.0,
              'sampler': 'random',
              'neg_sampler': 'in_degree',
              'temperature': 0.07
              }

    run(config)
```




到这里，快看！使用docker部署图深度学习框架GraphLearn使用说明 就写完了，有问题欢迎留言讨论哦～ 

---

宅男民工码字不易，你的关注是我持续输出的最大动力。

接下来作者会继续分享学习与工作中一些有用的、有意思的内容，点点手指头支持一下吧～

欢迎扫码关注作者的公众号： 算法全栈之路

![](https://gitee.com/ldh521/picgo/raw/master/2021-7-18/1626539300022-qrcode_for_gh_63df84028db0_258.jpg)






