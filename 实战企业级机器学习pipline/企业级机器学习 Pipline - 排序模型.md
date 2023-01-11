企业级机器学习 Pipline - 排序模型

---

在模型开篇介绍前，作者要再三阐述一个观点就是：没有所谓的简单模型或则复杂模型，只有最适合业务的模型。只要能解决业务问题，哪怕是使用最简单的模型，那也是非常优秀的模型。让合适的模型出现在合适的位置，深刻理解模型的底层原理与应用场景，一切为业务服务是算法工程师的根本宗旨。

书接上文，我们使用召回模型返回的大约1K左右的用户可能感兴趣的item集合，最终需要送入排序模型里面去进行排序。在推荐场景里，无论是推荐自然量还是推荐广告，现有的主流方法都会以**建模item的点击率或转化率**为目标。

---

### (1) 排序模型基础简介

当然，现在有很多公司也在尝试着learn to rank (LTR) 的方式，构建 **point wise**, **pair wise**, **list wise** 方式的各种排序模型。

但是在广告点击率预估场景中，还是以point wise 为主，得到需要排序的各个item 的排序因子，可以是点击率(CTR),也可以是转化率(CVR)等。毕竟广告算法场景，涉及到广告出价，需要计算Ecpm=ctr * bid, 而在这个公式中，点击率 ctr 是一个数值。

pair wise 建模的是需要排序list中两两item之间的相互比较的关系。假设使用三元组来表示pair wise模型的样本，如果(A,B ,+1) ,则(B,A,-1)。

list wise 建模的是整个列表的先后关系，让整个list损失最小。注意：pair wise 和 list wise 多用于自然量item 推荐的排序模块。

这里我们仅仅介绍point wise 形式的排序模型，如果对其他的LTR排序模型感兴趣的同学，可以留言加入讨论群进行交流～

![](https://img.soogif.com/IzqVKZUu3xCA8pqhXTaQvLMYhRnhEM79.gif?imageMogr2/thumbnail/!89.94296828548215p&scope=mdnice)

---

### (2)粗排与精排

在很多需要排序的业务场景里，排序模型又被更近一步分为粗排模型与精排模型(粗排并不是必须的，也有很多场景里只有精排模块)。

一般相对于精排模型，粗排会使用较少的特征、简单的模型来进行排序，业界很多公司在粗排阶段使用逻辑回归(LR)模型居多。在广告推荐的场景里，也会使用粗排模型predict得到ctr计算出粗排 `ecpm=ctr * bid` ，然后根据粗排的ecpm做一次截断进行粗步item选择，更近一步的减少进入精排模型的item数量，一般是从约1k减少到200以内。

注意：广告item的点击率常规意义上可以把理解为广告的质量，而bid表示着广告主愿意为这次曝光提供的出价，两者乘积综合影响着最终的排序。
在更复杂的广告排序场景里，排序策略会加入更多的排序因子如：

`ecpm= bid*ctr* cvr* channel_weight`

等，也有采用价格挤压等策略来进行因子调权，这里暂不进行展开，感兴趣的可以自己下去了解。

无论精排还是粗排模型，都属于排序模块。上文召回模块已经介绍过，召回模块将用户可能喜欢的，和海量对用户根本不靠谱的目标分隔开来;排序模块则是从用户可能喜欢的候选集当中挑选出用户最喜欢的，是为了优中选优。所以一般在排序模块，算法工程师们会设计大量的特征去刻画用户的兴趣，并使用复杂的模型去对用户兴趣进行建模。

这里强调特征与模型对排序模块的重要性，并不是说样本不重要，相反无论是召回和排序模型，样本都是和业务场景深度绑定且非常重要的。在前面的章节，对样本以及特征设计均进行了详细的介绍，这里也不再展开。这里以精排模型为例详细介绍。

![](https://img.soogif.com/H2ljc2eYd1LXsFPeJvYsCXtEYmO0a6r2.gif?scope=mdnice)

---

### (3) 精排模型

#### (3.1) 逻辑回归(LR/FTRL)

说到精排模型，LR模型以其解释性强、使用简单、线上部署轻量级、inference速度快等优点成为很多公司最初baseline模型的首选。

其中 2013年Google 提出了LR模型的变种模型FTRL，其在以高维稀疏特征为主要特征的排序场景一时风头无两，甚至直到2023年的今天依然再广告算法场景占有一席之地。这与模型特点也业务场景的独特性有着重要的关系，因为在广告算法场景，很多头部广告主的预算严重影响着公司的收入，而服务好这部分大预算的广告主就显得非常重要。

LR模型以其记忆性强的特性，相对于后来DNN模型的泛化性特性，偏记忆性的模型更突出了对头部广告主广告的偏爱，可以保证公司的收入持续稳定。所以一直到今天，LR模型在很多广告算法场景依然占据着稳稳一席之地。据我所知，小米应用商店和百度凤巢的某些比较重要的广告算法场景依然在使用LR 模型。

---

#### (3.2) WideDeep

另一个经典的模型是 Wide&Deep模型，也是Google团队在 DLRS 2016提出的。该模型是一种应用于推荐场景的深度学习模型,可以是推荐自然量，也可以是推荐广告，此模型包含了 wide 和 deep 两部分。

Wide部分可以是LR模型，也可以是FM模型，一般会输入大量的高维稀疏的特征，通常是大量的ID类、类别类特征，最后计算出属于wide部分的logit值。
Deeep部分一般是DNN模型，一般是将高维稀疏的特征转化为低维稠密的特征，通常是embeding，然后针对这些embeding进行pooling或则拼接操作，后面在接几层全连接层，最后计算出属于deep部分的logit值。

最后一层则是将wide和deep部分的logit值加起来扔进交叉熵 loss 里进行优化。

其中，这里要重点强调一下模型的结构。数据和模型可以说是 **一体两面** ，**一体** 就是指建模用户兴趣，而 **两面** 则是指我们可以为了更好的建模所使用的特征数据来进行不断调整模型结构，同时也可以将数据处理成某个格式来适配相应的模型。

对于高维稀疏的特征，我们可以把丢进wide侧，也可以把丢到deep求的embeding; 

而对于统计特征，我们可以先使用树模型,例如xgboost模型进行分桶然后在丢到widedeep模型里，也可以直接把统计特征的浮点数丢入deep侧和embeding contact(尝试多种效果不理想);

对于序列特征,我们可以使用`lstm ,din ,dien,multi head self attention` 进行建模;

对于实时特征，我们可以构建曝光序列，点击序列，搜索序列等，可以建模出时间信息然后丢入wide侧或则deep。

而对于拥有表格形式的大量稠密的统计特征的场景，直接使用 **树模型** 来建模分类和回归业务则是上佳的选择。

注意: 一般对于widedeep模型说来，wide 部分采用ftrl作为优化器，而deep部分则是无脑采用adam作为优化器。


我们都知道一个机器学习排序模型预估系统其实则对应着一整套机器学习pipline, 中间涉及了原始数据清洗，训练样本选择，特征设计，模型结构设计与训练，到最后的实时/离线 predict 的过程 。

因为模型的打分Score决定着给用户推荐Item的顺序，关系着用户体验与公司收入，所以predict 打分的准确性是至关重要的。上面所说的数据、样本、特征、模型等标准机器学习pipline里的各个模块均对模型的实验效果有着重要的影响。

一般我们会对一个模型进行持续性迭代优化，我们也可以从数据、样本、特征、模型等方面进行迭代，有时候一个微小的改变均能对用户某一个方面的特性进行更好的刻画，带来线上业务收入的重大提升。so 多实验多尝试，一切以线上收入提升为准。
![](https://img.soogif.com/XphuaA7i3KyFDJg0VqPv2sAK3PDRNORf.gif?scope=mdnice)


---

### (4)模型效果评估 

最后，对于模型打分效果评估，如果我们总是进行线上实验，时间成本和金钱成本都是非常巨大的。所以有一个和线上业务指标深度契合的离线实验观测指标也是非常重要的。

一般我们在会选择 `Auc、GAuc、LogLoss、Copc`，以及配对T检验来对离线以及线上观测指标进行观测，看我们的实验指标与baseline 模型效果上是否有显著性差异。其中我们也会 **持续观测一个模型的离线CTR(CVR)以及线上实时CTR(CVR)** ，来进行更进一步的分析。

注意： 我们进行离线实验的时候，有一个基本原则就是控制变量。当我们迭代样本的时候，保持特征、模型不变，而做特征实验的时候，则保持样本与模型不变。这样可以让我们知道实验效果的影响来自于哪个部分，无论效果好坏，这可以让我们的算法工程师对模型与业务有着更深刻的理解与掌控。


---

### (5)相关历史文章一锅端 

上面提到的一些知识，作者以前发表的文章里均有部分涉及，可以去这里查看：

[企业级机器学习 Pipline - log 数据处理](https://blog.csdn.net/qq_25459495/article/details/119845792)   `https://blog.csdn.net/qq_25459495/article/details/119845792`

[企业级机器学习 Pipline - 样本sample处理](https://blog.csdn.net/qq_25459495/article/details/119857180)  `https://blog.csdn.net/qq_25459495/article/details/119857180 `

[企业级机器学习 Pipline - 特征feature处理 - part 1](https://blog.csdn.net/qq_25459495/article/details/119984603)  `https://blog.csdn.net/qq_25459495/article/details/119984603`

[企业级机器学习 Pipline - 召回模型](https://blog.csdn.net/qq_25459495/article/details/128461465)   `https://blog.csdn.net/qq_25459495/article/details/128461465`

[算法工程师常用python脚本，这原理你真的理解透了吗？](https://zhuanlan.zhihu.com/p/405145251)  `https://zhuanlan.zhihu.com/p/405145251`

[算法工程师打死都要记住的20条常用shell命令](https://zhuanlan.zhihu.com/p/404589964)  `https://zhuanlan.zhihu.com/p/404589964`

---


### (6)代码时刻 


 talk is cheap , show me the code !!! 

下面的代码是使用tensorflow2.0,采用了tf.keras 中阶API来构建模型结构，使用了featureColumn 特征处理API来处理特征，具有极高的参考价值与可重用性，有问题欢迎讨论。

该工程完整代码，可以去算法全栈之路公众号回复“**排序模型源码**”下载。

---


```
# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse
import tensorflow as tf
import log_util
import params_conf
from date_helper import DateHelper
import data_consumer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
    x = tf.keras.layers.concatenate(define_list + common_list)

    # 可以在下面接入残差网络
    for i, dnn_hidden_size in enumerate(params_conf.DNN_HIDDEN_SIZES):  # DNN_HIDDEN_SIZES = [512, 128, 64]
        x = tf.keras.layers.Dense(dnn_hidden_size, activation="relu", name="overall_dense_%s" % i)(x)

    model_outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="final_sigmoid")(x)

    model = tf.keras.Model(
        inputs=inputs_list,
        outputs=model_outputs)

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
    log_util.info("get train data finish ...")

    emb_map, feature_inputs_list = build_embeding()
    log_util.info("build embeding finish...")

    # 构建模型
    model = build_model(emb_map, feature_inputs_list)
    log_util.info("build model finish...")

    adam_optimizer = tf.keras.optimizers.Adam(params_conf.LEARNING_RATE)
    model.compile(
        optimizer=adam_optimizer,
        loss=params_conf.LOSS,  # LOSS = "binary_crossentropy"
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
    model.fit(
        train_set,
        validation_data=test_set,
        epochs=params_conf.NUM_EPOCHS,  # NUM_EPOCHS = 10
        steps_per_epoch=params_conf.STEPS_PER_EPHCH,
        validation_steps=params_conf.VALIDATION_STEPS,
        #
        callbacks=[tensorboard_callback, lr_schedule_callback]
    )

    # 模型保存
    tf.keras.models.save_model(model, model_full_output_dir)

    # tf.saved_model.save(model, model_full_output_dir)
    print("save saved_model success")


def test():
    output_root_dir = "{}/{}/{}".format(params_conf.BASE_DIR, args.model_output_dir, args.cur_date)
    model_full_input_dir = os.path.join(output_root_dir, "model_savedmodel")

    tf.keras.backend.clear_session()
    log_util.info("start test...")
    # 此处多跑一天,把-1带上,防止读数据报错
    train_date_list = DateHelper.get_date_range(DateHelper.get_date(-1, args.cur_date),
                                                DateHelper.get_date(0, args.cur_date))
    train_date_list.reverse()
    print("train_date_list:" + ",".join(train_date_list))

    # load data from tf.data,兼容csv 和 tf_record
    # 仅仅考虑测试集
    _, test_set = data_consumer.get_dataset(args.train_data_dir, train_date_list,
                                            get_feature_column_map().values())
    log_util.info("get test data finish ...")

    model = tf.keras.models.load_model(model_full_input_dir)
    print("restore saved_model success")

    model.evaluate(test_set, batch_size=params_conf.BATCH_SIZE)


if __name__ == "__main__":
    print(tf.__version__)

    # run tensorboard:
    # tensorboard --port=8008 --host=localhost --logdir=../log
    args = init_args()
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()

```

---


码字不易，觉得有收获就点赞、分享、再看三连吧~

欢迎扫码关注作者的公众号： 算法全栈之路

![](https://gitee.com/ldh521/picgo/raw/master/2021-7-18/1626539300022-qrcode_for_gh_63df84028db0_258.jpg)



