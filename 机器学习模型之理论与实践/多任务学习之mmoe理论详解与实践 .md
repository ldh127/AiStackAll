多任务学习之mmoe理论详解与实践 

---

书接上文，在前一篇文章 [快看 esmm 模型理论与实践](https://zhuanlan.zhihu.com/p/597004211) 中，我们讲到MTL任务通常可以把可以分为两种: **串行与并行** 。多个任务之间有较强关联的, 例如点击率与转化率，这一种通常我们可以使用 ESMM 这种串行的任务进行 **关系递进性与相关性** 建模。而对于多个任务之间相对比较独立的，例如点击率与用户是否给出评论的评论率，通常可以选择 MMOE 这种并行的任务进行 **相关性与冲突性** 的建模。

**MMOE** 模型全称是 **Multi-gate Mixture-of-Experts**,  该模型由 Google在 2018年 KDD 上发表的文章 Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts 中提出的。

MMOE 模型本身在结构上借鉴了以前的 MOE 模型，但又有一定的创新, 它可以说是提出了一种新的 **MTL(Multi-Task Learning)** 架构，对每个顶层任务均使用了一个 gate网络 去学习融合多个专家对当前 Task 的权重影响， 在很大程度上调节缓解 多目标任务相关性低导致的准确率低 的问题。

本文，我们主要对mmoe模型进行理论与实践过程的阐述...

---
### (1) mmoe和esmm模型的对比

说到 **mmoe** 模型，我们一般总是会把它**横向**的和阿里巴巴提出的**esmm模型进行对比**，而**纵向**的和最初**shared-bottom、moe**进行对比，这里我们分别也从横向和纵向对比展开。

首先是 **esmm模型** ，和mmoe一样 ,  它也是一种(Multi-Task Learning,MTL) 模型。esmm 模型相关内容，可以去这里  [快看 esmm 模型理论与实践](https://zhuanlan.zhihu.com/p/597004211)  查看。但是上文我们也说了, esmm 我们建模的是一种串行的任务，任务关系之间有 递进 的性质。而mmoe则不要求两个任务有递进的关系。

在这里，我们之所以一直强调esmm模型适合建模递进任务，我们可以从esmm的损失函数可以看到：
![](https://files.mdnice.com/user/17436/4c695f72-bde8-4a48-a011-8515fea8eaf2.jpg)
, 我们可以重点关注下他的CTCVR损失。

而建模的pCTCVR 的 概率建模逻辑 又可以这样表示：
![](https://files.mdnice.com/user/17436/e2a0000d-9e24-41fe-ae76-50ae0ca4f4de.jpg)

读过上面介绍esmm文章的同学就可以看到，pCTCVR 任务的核心是： **我们站在曝光的前提下，不知道点击与否，不知道转化与否，这个时候去预估点击并且转化的概率**。

如上所述，CTR 任务和 CVR任务 都是进行分类模型的0/1概率预估, 用到的**特征相似, 场景也相似, 任务相关性非常高**。并且，该模型使用的业务场景也要求了在曝光后，**item先被点击才会有转化，没有被点击就压根谈不上转化了，这是一种严格的递进关系**。同时**CTR与CVR两者的概率相乘有着明显且理论正确的业务意义**。


我们从 上篇文章 里也了解到:  esmm 模型从任务的根本“损失函数”上就保证了这种曝光前就去预估点击且转化的概率建模逻辑。  如果任务本身没有 **严格递进且高度相似相关**，那我们这里的损失建模就没有意义了。

或则，我们也可以改变esmm模型的损失函数(去掉严格递进且高度相似相关的关系逻辑)，用它去建模一些不那么相关的 multi-task 任务，毕竟是DNN模型，只要你能把跑起来，总是可以predict出一个结果的。但是如果损失函数有了大的本质上的改变，esmm模型就失去了它的精华，那这个模型还叫esmm模型吗？


而对于本文介绍的**MMOE模型**，我们对他的样本就没有那么多的要求，它可以进行并行与冲突性建模。当然，没有那么多的要求不是没有要求，至少两个任务的样本和特征是能共享的吧，业务能够有交叉的吧，最重要的是多个任务的损失融合后能找到一定的优化意义的吧。

例如：**视频推荐**中，我们有多个不同甚至可能发生冲突的目标，就像多个task是去预估用户是否会观看的同时还希望去预估用户的观看时长，同时预估用户对于视频的评分。这中间就同时有 **分类任务** 和 **回归任务** 了。

这里我们引入一些 **任务相关性衡量** 的简单介绍： 

我们知道: **Spearman相关系数仅评估单调关系，Pearson相关系数仅评估线性关系**。例如： 如果关系是一个变量在另一个变量增加时增加，但数量不一致，则Pearson相关系数为正但小于+1。 在这种情况下，Spearman系数仍然等于+1。

我们这里说的 **两个任务的相关性，可以通过 两个任务的label 之间的 皮尔逊相关系数 来表示** 。假设模型中包含两个回归任务，而数据通过采样生成，并且规定输入相同，输出label不同，求得他们的皮尔逊相关系数，相关系数越大，表示任务之间越相关，相关系数越大，表示任务之间越相关。

---

### (2)  mmoe 模型详解

说到 mmoe，因其一脉相承血浓于水，不得不提到 shared-bottom与moe 这两个模型。这里，我们通过对比三种模型的 异同 来 逐渐 引出 mmoe 模型的特点。
闲言少叙，上图：
![](https://files.mdnice.com/user/17436/c404875e-7655-4900-9ac9-e2e64fc2cc2f.JPG)


从上面图中，我们可以看出：在三个图(a,b,c)中，从下往上分别是模型的输入到输出的流程。

如上图a所示，假设我们的模型中有N个任务，则在上层会有K个塔 (图中K=2)。其中，**上层的每个塔对应一个特定任务的个性化知识学习，而底层的shared-bottom 层作为共享层，可以进行多个任务的知识迁移**。

**注意**：输入的input在我们的理解里，已经是各个sparse 或dense特征得到embeding 并且拼接之后的结果，理论上是一个[batch_size, embeding_concat_size]的tensor。

---

#### （2.1）原始的shared-bottom 模型

如上图所示：**图a** 是原始的 **shared-bottom 结构**，2个 task 共享隐藏层的输出结果, 输入到各自任务的tower,  **一般的 tower 也就是几层 全连接网络** ，最后一层就是分类层或回归层，和常规的分类回归任务没有什么区别。

---

#### （2.2） moe网络 

**图b**  就是最初版本的 **moe 网络** ，我们从图中可以看到有个gate网络，并且只有一个gate门控网络，同时图b中有三个专家网络。

我们可以看到：**gate门控网络的输出维度和专家个数相同，起到了融合多个专家知识的作用**。融合完了之后会有一个公共的输出，该相同的输出分别输入到上面2个tower中。

常规网络中，gate网络和tower网络均是几层 全连接网络，只是最后的输出看情况考虑维度以及是否需要添加 relu 与 softmax 函数等。


这里要**注意**一点就是： **我们的input是分别作为各个专家和gate门控网络的输入**，各个专家和门控网络分别独自初始化以及训练。gate网络的输出的各个维度和各个专家的输出进行加权求和，得到一个综合专家维度的输出，然后相同的输入分别输入到上面两个不同的任务中。

这里我们从网络结构上可以明显看到**MOE和初始网络**的区别，多了一个多专家加权融合的门控网络，使得各个专家学习到的知识进行平滑的融合，可以让模型训练的更好。

---

#### （2.3）mmoe网络的提出

**图c**  就是我们本文要重点介绍的 **mmoe 网络**了。
mmoe 说白了，就是一种新的MTL网络架构的创新。mmoe **实际上就是 多个门 的 moe 网络**。
输入多个专家的过程和moe无任何区别，这里唯一的不同是对**每一个任务有一个门控网络**。

mmoe和moe最大的差别就在于输入上面任务的输入。
**moe的任务tower输入的是经过同一个门控网络加权过的多个专家的输出，是相同的一个embeding。
 而mmoe 的任务tower 输入的则是经过自己任务特有的门控网络加权过的多个专家的输出，对于不同任务是不同的**。没有明显增加参数，却对网络的学习起到了重要的影响作用。

**通俗理解** ： 我们网络中的每个专家都是可以学到一些关于**多个任务的各自的专业知识**，而我们用多个门控网络，就相当于起到了一个**Attention**的作用。就例如： 我们使用一个多目标任务网络去预估一个人分别得感冒和高血压的概率，我们现在有多个专家都会相同的这个病人进行会诊，但是每个专家各有所长又各有所短，这个时候，我们就通过一个门控网络去自动的学习对于某种病情应该多听从哪个专家的意见，最后对各个专家的意见进行加权求和之后来综合评定这个人患某种病的概率。

让每个专家发挥出各自的特长，是不是更有利于我们实际的情况呢？

而开篇所提到的**mmoe网络的 冲突性建模能力 也就来自于这多个门控网络对于多个任务可学习的调控能力**。**多个专家加上多个门控网络，不同任务对应的门控网络可以学习到不同的 Experts 组合模式，模型更容易捕捉到子任务间的相关性和差异性，能够使得我们多个任务融合的更加平滑** ，最终打分得出的结果也更加能够动态综合多个专家的特长与能力，得出一个更有益于我们业务目标的结果。 

前文我们已经介绍过， mmoe网络的提出主要就是提出了一个新的MTL架构。所以上文中，我就没有在引入一些晦涩难懂的公式，而是全部采用了文字说明的形式来下进行阐述，希望能似的读者看起来更丝滑一些～ 

---

### (3) mmoe 模型实践与心得

其实相对于esmm模型，mmoe模型更好理解，构造样本等也更加容易。

但是仍然有一点就是： 我们在使用**tensorflow 或 pytorch 实现网络**的过程中，多个专家以及门控网络的输入输出维度对应上有一定难度，有隐藏的暗坑在里面。不过这些在上文中，我也大概以文字的形式说清楚了，后面分享的代码源码我也根据自己的理解进行了详细的注释，希望对读者的理解有帮助哈～ 

在实际使用mmoe的过程中，有同学会遇到：**训练mmoe的过程中，发现多个gate的输出过度偏差** ，例如：（0.999,0.001)情况。 
这一种情况初步感觉还是：**(1)** **网络的实现**有问题，需要去排查下各个专家网络以及门控网络的初始化值有没有问题。 **(2)** 去**排查下两个任务的标签**情况，是不是两任务的标签呈现比较多的极端情况，也可以采用上面介绍的任务相关性衡量办法看一下两个任务的相关性。 在输入相同的情况下在网络理论上不应该出现这个问题我在使用过程中并没有遇到，所以只能给出一些猜测的解决方法...


当我们遇到两个任务的**目标差异特别巨大**时，例如：预估视频点击率与观看时长。这个任务我们应该直觉上就觉得标签的差异太过于大了，时长的label最好能够进行一定的处理。例如log处理。

log函数有着优秀的性质，经过**log处理**后目标会削弱量级，数据也会变得更符合正态分布，同时**loss也更小和稳定** 。loss的稳定对于多任务模型学习和训练来说是至关重要的，它影响着**多个任务根据loss更新的梯度，最好我们能够把多个目标的loss加权重调到同一量级**，对这种差异比较大的问题总是能够起到缓解作用的额


同时在进行**mmoe网络设计**的过程中，我们不仅可以使用多个任务有**共享的专家(官方版本)**，其实我们也可以给每个任务加上**各自独特的专家**进行组合学习，期望模型可以学习到各个任务之间的**个性与共性**。

另外, 我们可以将mmoe 作为一种复杂的DNN layer ,我们可以在网络中**叠加多个 mmoe layer** , 实现一些比较复杂的网络来学习一些比较复杂的多目标任务。 

---

### (4) 代码时光 

**talk is cheap , show me the code !!!**

哎， 终于再次写到代码时光了！ 

对于mmoe 模型，才开始看源码到最后理解花了挺长时间，中间主要的时间都花在了实现的时候专家网络和多门控网络的输入输出维度对应上。下面的代码注释均写的比较详细，看的过程中，如有任何问题欢迎公众号留言讨论 ～


```

@ 欢迎关注微信公众号：算法全栈之路
# coding:utf-8

import numpy as np
import os
import argparse
import tensorflow as tf
import log_util
import params_conf
from date_helper import DateHelper
import data_consumer
from mmoe import MMoE
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import VarianceScaling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import tensordot, expand_dims
from tensorflow.keras import layers, Model, initializers, regularizers, activations, constraints, Input

from tensorflow.keras.backend import expand_dims, repeat_elements, sum


class MMoE(layers.Layer):
    """
    Multi-gate Mixture-of-Experts model.
    """

    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: Number of hidden units 隐藏单元
        :param num_experts: Number of experts 专家个数,可以有共享专家,也可以有每个任务独立的专家
        :param num_tasks: Number of tasks  任务个数,和tower个数一致
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights. 专家的权重是否添加偏置
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights. 门控的权重是否添加偏置
        :param expert_activation: Activation function of the expert weights.  专家激活函数
        :param gate_activation: Activation function of the gate weights.  门控激活函数
        :param expert_bias_initializer: Initializer for the expert bias. 专家偏置初始化
        :param gate_bias_initializer: Initializer for the gate bias. 门控偏置初始化
        :param expert_bias_regularizer: Regularizer for the expert bias. 专家正则化
        :param gate_bias_regularizer: Regularizer for the gate bias.  门控正则化
        :param expert_bias_constraint: Constraint for the expert bias. 专家偏置
        :param gate_bias_constraint: Constraint for the gate bias.  门控偏置
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class  附属参数若干
        """
        super(MMoE, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # Activation parameter
        # self.expert_activation = activations.get(expert_activation)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []
        # 在初始化的过程中,先构建好网络结构
        for i in range(self.num_experts):
            # 有几个专家, 这里就添加几个dense层, dense层的输入为上面传入, 当前层的输出维度为units的值, 隐藏单元个数
            self.expert_layers.append(layers.Dense(self.units, activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   bias_initializer=self.expert_bias_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))

        # 门控网络, 门控网络的个数等于任务数目 , 但是取值数据的维度等于专家个数 , mmoe 对每个任务都要融合各个专家的意见。
        # 有几个任务,
        for i in range(self.num_tasks):
            # num_tasks个门控,num_experts维数据
            self.gate_layers.append(layers.Dense(self.num_experts, activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))

    def call(self, inputs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        # assert input_shape is not None and len(input_shape) >= 2

        # 三个输出的网络
        expert_outputs, gate_outputs, final_outputs = [], [], []

        # 专家网络
        # 有几个专家循环几次
        for expert_layer in self.expert_layers:
            # 注意这里是当前专家的变化
            # 输入的元素元素应该是整体embeding contact之后的一堆浮点数维度数据。
            # (batch_size, embedding_size,1)
            expert_output = expand_dims(expert_layer(inputs), axis=2)
            # nums_expert * ( batch_size, unit, 1)
            expert_outputs.append(expert_output)

        # 同 batch 的数据,既然是沿着第一个维度对接,那根本就不用看第二个维度,那个axis的维度数目相加
        #  nums_expert * ( batch_size, unit, 1) -> 这里 contact 之后,列表里 num_experts 个 tensor 在最后一个维度concat到一起,
        # 则最后维度变成了 ( batch_size, unit, nums_expert ),只有最后一个维度的维度值改变了。
        expert_outputs = tf.concat(expert_outputs, 2)

        # 门控网络, 每个门对每个专家均有一个分布函数.
        for gate_layer in self.gate_layers:
            # 对于当前门控,[ batch_size,num_units ] ->  [ nums_expert,batch_size,num_units ]
            # 有多少个任务,就有多少个gate
            # num_task * (batch_size,num_experts),这里对每个专家只有一个数值,和专家的输出维度unit相乘需要拓展维度
            gate_outputs.append(gate_layer(inputs))

        # 这里每个门控对所有的专家进行加权求和
        for gate_output in gate_outputs:
            # 对当前gate,忽略 num_task维度,为 (batch_size, 1, num_experts)
            expanded_gate_output = expand_dims(gate_output, axis=1)
            # 每个专家的输出和gate的数据维度相乘
            # ( batch_size, unit, nums_expert ) *  (batch_size, 1 * units, num_experts),因此 1*units
            # If x has shape (s1, s2, s3) and axis is 1, the output will have shape (s1, s2 * rep, s3).
            # 这里的本质是 门控和专家的输出相乘维度不对,如上面所说,门控维度1和需要拓展到各个专家的输出维度 unit,方便相乘。
            # "*"算子在tensorflow中表示element-wise product，即哈达马积,即两个向量按元素一个一个相乘，组成一个新的向量，结果向量与原向量尺寸相同。
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, self.units, axis=1)

            # 上面输出的维度是 (batch_size, unit, nums_expert ),对第二维nums_expert求和则该维度就变成一个数值 -> (batch_size,unit)
            # 这里对各个专家的结果聚合之后,返回的是一个综合专家对应的输出单元unit维度.
            # 最终有多个门控,上面多个塔,这里返回的是 num_tasks * batch * units 这个维度。
            final_outputs.append(sum(weighted_expert_output, axis=2))

        # 返回的矩阵维度 num_tasks * batch * units
        # 返回多个门控,每个门控有综合多个专家返回的维度 units
        # 这里 final_outputs返回的是个list,元素个数等于 门控个数也等于任务个数
        return final_outputs


def init_args():
    parser = argparse.ArgumentParser(description='dnn_demo')
    parser.add_argument("--mode", default="train")
    parser.add_argument("--train_data_dir")
    parser.add_argument("--model_output_dir")
    parser.add_argument("--cur_date")
    parser.add_argument("--log", default="../log/tensorboard")
    parser.add_argument('--use_gpu', default=False, type=bool)
    args = parser.parse_args()
    return args


def get_feature_column_map():
    key_hash_size_map = {
        "adid": 10000,
        "site_id": 10000,
        "site_domain": 10000,
        "site_category": 10000,
        "app_id": 10000,
        "app_domain": 10000,
        "app_category": 1000,
        "device_id": 1000,
        "device_ip": 10000,
        "device_type": 10,
        "device_conn_type": 10,
    }

    feature_column_map = dict()
    for key, value in key_hash_size_map.items():
        feature_column_map.update({key: tf.feature_column.categorical_column_with_hash_bucket(
            key, hash_bucket_size=value, dtype=tf.string)})

    return feature_column_map


def build_embeding():
    feature_map = get_feature_column_map()
    feature_inputs_list = []

    def get_field_emb(categorical_col_key, emb_size=16, input_shape=(1,)):
        # print(categorical_col_key)
        embed_col = tf.feature_column.embedding_column(feature_map[categorical_col_key], emb_size)
        # 层名字不可以相同,不然会报错
        dense_feature_layer = tf.keras.layers.DenseFeatures(embed_col, name=categorical_col_key + "_emb2dense")
        feature_layer_inputs = dict()

        # input和DenseFeatures必须要用dict来存和联合使用，深坑啊！！
        feature_layer_inputs[categorical_col_key] = tf.keras.Input(shape=(1,), dtype=tf.dtypes.string,
                                                                   name=categorical_col_key)
        # 保存供 model input 使用.
        feature_inputs_list.append(feature_layer_inputs[categorical_col_key])
        return dense_feature_layer(feature_layer_inputs)

    embeding_map = {}
    for key, value in feature_map.items():
        # print("key:" + key)
        embeding_map.update({key: get_field_emb(key)})

    return embeding_map, feature_inputs_list


def build_dnn_net(net, params_conf, name="ctr"):
    # 可以在下面接入残差网络
    for i, dnn_hidden_size in enumerate(params_conf.DNN_HIDDEN_SIZES):  # DNN_HIDDEN_SIZES = [512, 128, 64]
        net = tf.keras.layers.Dense(dnn_hidden_size, activation="relu", name="overall_dense_%s_%s" % (i, name))(net)
    return net


def build_model(emb_map, inputs_list):
    # 需要特殊处理和交叉的特征,以及需要短接残差的特征,可以单独拿出来
    define_list = []
    adid_emb = emb_map["adid"]
    device_id_emd = emb_map["device_id"]
    ad_x_device = tf.multiply(adid_emb, device_id_emd)

    define_list.append(ad_x_device)

    # 直接可以拼接的特征
    common_list = []
    for key, value in emb_map.items():
        common_list.append(value)

    # embeding contact
    net = tf.keras.layers.concatenate(define_list + common_list)

    # Set up MMoE layer
    # 返回的矩阵维度 num_tasks * batch * units
    # 返回多个门控,每个门控有综合多个专家返回的维度 units
    # 这里 final_outputs返回的是个list,元素个数等于 门控个数也等于任务个数
    mmoe_layers = MMoE(units=4, num_experts=8, num_tasks=2)(net)

    output_layers = []

    # Build tower layer from MMoE layer
    # 对每个 mmoe layer, 后面均接着 2层dense 到输出,
    # list,元素个数等于 门控个数也等于任务个数
    for index, task_layer in enumerate(mmoe_layers):
        # 对当前task, batch * units 维度的数据, 介入隐藏层
        tower_layer = layers.Dense(units=8, activation='relu', kernel_initializer=VarianceScaling())(task_layer)
        # 这里unit为1,当前任务为2分类
        output_layer = layers.Dense(units=1, name="task_%s" % (index), activation='sigmoid',
                                    kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    # Compile model
    # 这里定义了模型骨架,input 为模型输入参数,而output_layers 是一个列表,列表里返回了2个任务各自的logit
    # 其实分别返回了每个task的logit,logit这里为分类数目维度的数组,2维过softmax

    model = Model(inputs=[inputs_list], outputs=output_layers)

    return model


def train():
    output_root_dir = "{}/{}/{}".format(params_conf.BASE_DIR, args.model_output_dir, args.cur_date)
    os.mkdir(output_root_dir)
    model_full_output_dir = os.path.join(output_root_dir, "model_savedmodel")
    # print info log
    log_util.info("model_output_dir: %s" % model_full_output_dir)

    # 重置keras的状态
    tf.keras.backend.clear_session()
    log_util.info("start train...")
    train_date_list = DateHelper.get_date_range(DateHelper.get_date(-1, args.cur_date),
                                                DateHelper.get_date(0, args.cur_date))
    train_date_list.reverse()
    print("train_date_list:" + ",".join(train_date_list))

    # load data from tf.data,兼容csv 和 tf_record
    train_set, test_set = data_consumer.get_dataset(args.train_data_dir, train_date_list,
                                                    get_feature_column_map().values())
    # train_x, train_y = train_set

    log_util.info("get train data finish ...")

    emb_map, feature_inputs_list = build_embeding()
    log_util.info("build embeding finish...")

    # 构建模型
    model = build_model(emb_map, feature_inputs_list)
    log_util.info("build model finish...")

    def my_sparse_categorical_crossentropy(y_true, y_pred):
        return tf.keras.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    opt = tf.keras.optimizers.Adam(params_conf.LEARNING_RATE)

    # 注意这里设定了2个损失分别对应[ctr_pred, ctcvr_pred] 这两个任务
    # loss_weights=[1.0, 1.0]这种方式可以固定的调整2个任务的loss权重。
    model.compile(
        optimizer=opt,
        loss={'task_0': 'binary_crossentropy', 'task_1': 'binary_crossentropy'},
        loss_weights=[1.0, 1.0],
        metrics=[
            tf.keras.metrics.AUC(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()]
    )
    model.summary()
    # tf.keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True, dpi=150)

    print("start training")

    # 需要设置profile_batch=0，tensorboard页面才会一直保持更新
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.log,
        histogram_freq=1,
        write_graph=True,
        update_freq=params_conf.BATCH_SIZE * 200,
        embeddings_freq=1,
        profile_batch=0)

    # 定义衰减式学习率
    class LearningRateExponentialDecay:

        def __init__(self, initial_learning_rate, decay_epochs, decay_rate):
            self.initial_learning_rate = initial_learning_rate
            self.decay_epochs = decay_epochs
            self.decay_rate = decay_rate

        def __call__(self, epoch):
            dtype = type(self.initial_learning_rate)
            decay_epochs = np.array(self.decay_epochs).astype(dtype)
            decay_rate = np.array(self.decay_rate).astype(dtype)
            epoch = np.array(epoch).astype(dtype)
            p = epoch / decay_epochs
            lr = self.initial_learning_rate * np.power(decay_rate, p)
            return lr

    lr_schedule = LearningRateExponentialDecay(
        params_conf.INIT_LR, params_conf.LR_DECAY_EPOCHS, params_conf.LR_DECAY_RATE)

    # 该回调函数是学习率调度器

    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

    # 训练
    # 注意这里的train_set 可以使用for循环迭代,tf 2.0默认支持eager模式
    # 这里的train_set 包含两部分,第一部分是feature,第二部分是label ( click, click & conversion)
    # 注意这里是 feature,(click, click & conversion),第二项是tuple,不能是数组或列表[],不然报数据维度不对,坑死爹了。
    model.fit(
        train_set,
        # train_set["labels"],
        # validation_data=test_set,
        epochs=params_conf.NUM_EPOCHS,  # NUM_EPOCHS = 10
        steps_per_epoch=params_conf.STEPS_PER_EPHCH,
        # validation_steps=params_conf.VALIDATION_STEPS,
        #
        # callbacks=[tensorboard_callback, lr_schedule_callback]
    )

    # 模型保存
    tf.keras.models.save_model(model, model_full_output_dir)

    # tf.saved_model.save(model, model_full_output_dir)
    print("save saved_model success")


if __name__ == "__main__":
    print(tf.__version__)
    tf.compat.v1.disable_eager_execution()

    # run tensorboard:
    # tensorboard --port=8008 --host=localhost --logdir=../log
    args = init_args()
    if args.mode == "train":
        train()
```


到这里，多任务学习之mmoe理论详解与实践就写完成了，欢迎留言交流 ～

---

宅男民工码字不易，你的关注是我持续输出的最大动力！！！

接下来作者会继续分享学习与工作中一些有用的、有意思的内容，点点手指头支持一下吧～

欢迎扫码关注作者的公众号： 算法全栈之路

![](https://gitee.com/ldh521/picgo/raw/master/2021-7-18/1626539300022-qrcode_for_gh_63df84028db0_258.jpg)


