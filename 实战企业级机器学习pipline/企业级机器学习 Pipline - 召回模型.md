企业级机器学习 Pipline - 召回模型

---

part0 
书接上文，我们介绍了**log数据处理**，**样本处理**，**特征处理**，接下来我们开始介绍 **模型训练** 相关的内容。这里的模型通常就是指机器学习模型。


在一个成熟的推荐系统里，我们的机器学习模型一般会作用于2个模块：召回与排序。在排序模块有粗排序，精排，重排等阶段的区分，在这里暂不展开叙述。在召回模块，很多公司一般都会有多路算法召回，比较经典的有双塔召回，协同过滤召回等，在某些业务也会使用基于统计策略的热度召回，分模块召回等。

---

### 1.初步介绍

召回，顾名思义就是说从大量的候选集(广告或自然量候选集)初步选择一些用户潜在可能感兴趣的item集候选集，在下一步里进入排序模型。所以召回的这个集合量级一般不会很大，通常都在1K以内。召回模型通常面对的是整个item 集合，所以我们要求召回模型既要尽可能多的保留相关性高的结果，又要保证速度，召回结果的好坏对整个推荐结果有着至关重要的影响。使用 基于深度学习推荐模型+高性能的近似检索算法可以说是现在业界通用的选择，这套方案也被称为 DeepMatch。
注意：很多候选集比较小的场景，例如很多公司某个app位置的广告候选集，本来全量候选集可能也就几百，也就没必要使用召回模块了，直接把全量item丢到排序模块就ok。

---

### 2.样本选择

模型是对数据集合分布的学习，所以采用什么样的数据来训练模型是至关重要的。在业界有一个通用的认知是："排序是特征的艺术，而召回则是样本的艺术，特别是负样本的艺术", 也有一种说法是说"负样本为王"。这均是为了说明样本选择对于召回模型的重要性。

一般我们训练排序模型，例如训练预估点击率排序模型，使用曝光且点击作为正样本，曝光未点击作为负样本，美团也有使用 above click的作法，即只拿点击item以上的未点击文章做负样本，并且这种选取样本的做法在线上确实均获得明显的收益。
很多人在做召回模型的时候，直接套用排序模型的样本选择思路，离线发现指标虽有一定的增长，但是线上死活就是没有效果，最后有重头逐阶段的定位问题出现在哪里，一般这种情况，很大概率是因为召回模型的样本选择不合理导致的。

究其根本原因，这是因为违反了机器学习的一条基本原则，就是 离线训练数据的分布应该与线上实际应用的数据保持一致。上面我们谈到召回与排序的不同，说到召回的候选集更大，所以要求速度更快。另一个更深层次的不同是：排序其目标是“从用户可能喜欢的当中挑选出用户最喜欢的”，是为了优中选优。而召回是“是将用户可能喜欢的，和海量对用户根本不靠谱的分隔开”，所以召回在线上所面对的数据环境是巨量且混杂的。
所以，要求合理的训练召回模型的样本，既要让模型知道什么是好，也要让模型知道什么叫不好，只有让模型见识够了足够多的不好，才能让模型达到" 开眼界、见世面 "的目的，从而在“大是大非”上不犯错误。

知乎上一个叫石塔西的同学这个地方解答的非常好。对于召回来说，正样本就是<user,doc>最匹配的，没有异议，就是用户点击的样本，<user,doc>最不靠谱的，是“曝光未点击”样本吗？ 这里牵扯到一个推荐系统里常见的bias，就是我们从线上日志获得的训练样本，已经是上一版本的召回、粗排、精排替用户筛选过的，即能够获得曝光的已经是对用户“ 比较靠谱 ”的样本了。拿这样的样本训练出来的模型做召回， 一叶障目，只见树木，不见森林 。
所以一个比较好的做法就是拿点击样本做正样本，拿随机采样做负样本。因为线上召回时，候选库里大多数的物料是与用户八杆子打不着的，随机抽样能够很好地模拟这一分布。

---

### 3.样本精细化考虑

同时对于热门物料，facebook的双塔召回样本处理方法中，对于热门物料做正样本的时候，采取了降采样，限制热门物料的样本条数; 当热门物料做负样本时，要适当过采样，抵销热门物料对正样本集的绑架；同时，也要保证冷门物料在负样本集中有出现的机会。

我们选择样本的时候，使用可以明显区分的正负样本来训练模型，在初期可以拿到明显的效果，但是在后期逐渐深入的时候会发觉模型缺乏更细粒度的区分能力，即只能学到粗粒度上的差异，却无法感知到细微差别。这个时候负样本的威力就体现出来了，算法工程师一般在训练模型的时候，添加一些Hard Negative增强样本，刻意让模型去有意识的关注一些更难区分的样本。
如何选取hard negative，业界有不同的做法。Airbnb在《Real-time Personalization using Embeddings for Search Ranking at Airbnb》一文中的做法，就是根据业务逻辑来选取hard negative:
(1) 增加与正样本同城的房间作为负样本，增强了正负样本在地域上的相似性，加大了模型的学习难度
(2 ) 增加“被房主拒绝”作为负样本，增强了正负样本在“匹配用户兴趣爱好”上的相似性，加大了模型的学习难度。

一般，样本的选择都是与我们应用模型的业务场景紧密相关的，这里只是提供一个简单的思路，具体的样本选择算法工程师在各自的业务场景里可以自由发挥，好比去除无效曝光，去除作弊流量，选择skip above等各种方法，以及上面提出来的hard negative 等方式。



---

### 4.双塔召回原理与模型

   这里介绍下业界通用的双塔召回模型。所谓双塔，就是指一个用户塔，一个item塔。

采用上面所说的样本，以及前面章节介绍的特征数据训练好模型之后，双塔模型如何使用呢？ 一般我们离线训练的时候导出各个item对应的塔的embeding存到redis或则别的基于内存的数据库中，而在线上的时候把用户侧和上下文侧特征把用户侧模型走一遍，实时预估出用户侧的embeding ,线上的时候再把得到的两个embeding(一个查缓存得到，一个实时预估得到)求cos相似度，根据相似度来选择距离最近的若干item返回。

因为用户侧和上下文侧特征每个请求都是不同的，但是需要比较的embeding只有一个，而对应的item侧的embeding则是整个候选集，量级非常大，所以一般我们离线将item侧的embeding灌库，通常使用FAISS(facebook ai Similarity Search)库，根据用户embeding 去item库里检索若干个距离最近的item返回。

在item塔侧，我们可以可以加入itemID，类别id,一级/二级类别等以及其他的与item相关的side information 特征。

在user塔侧，我们可以加入用户的基础画像特征，用户的历史行为画像以及用户的实时画像特征，同时我们也在用户侧塔里加一些当前流量相关的环境上下文特征与设备上下文相关的特征。其中，我们可以加入一些用户历史行为序列特征
，采用 self attention 或则 din ,dien attention等方式来进行处理。

模型的选择都是非常灵活的，各种最新的论文上的方法都可以进行尝试，融合的到一个召回模型里都是可以的。


talk is cheap , show me the code !!! 

下面的代码是使用tensorflow2.0,采用了tf.keras 中阶API来构建模型结构，使用了featureColumn 特征处理API来处理特征，具有极高的参考价值与可重用性，有问题欢迎讨论。

该工程完整代码，可以去微信公众号 **算法全栈之路** 回复 "**双塔召回源码**" 获得。


```

# 欢迎关注微信公众号： 算法全栈之路 
# -*- coding: utf-8 -*-

import traceback
import codecs
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
    parser.add_argument("--item_embeding_dir")
    parser.add_argument("--user_embeding_dir")
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
    feature_inputs_map = dict()

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
        feature_inputs_map[categorical_col_key] = feature_layer_inputs[categorical_col_key]
        return dense_feature_layer(feature_layer_inputs)

    embeding_map = {}
    for key, value in feature_map.items():
        embeding_map.update({key: get_field_emb(key)})

    return embeding_map, feature_inputs_map


def build_model(emb_map, inputs_map):
    # 分别单独拿出来ad侧和用户侧的特征,构建2个塔

    # 广告塔
    adid_emb = emb_map["adid"]
    site_id = emb_map["site_id"]
    site_domain = emb_map["site_domain"]
    site_category = emb_map["site_category"]
    app_id = emb_map["app_id"]
    app_domain = emb_map["app_domain"]
    app_category = emb_map["app_category"]
    ad_input = [inputs_map["adid"], inputs_map["site_id"], inputs_map["site_domain"], inputs_map["site_category"],
                inputs_map["app_id"], inputs_map["app_domain"], inputs_map["app_category"]]

    ad_stack = tf.keras.layers.concatenate(
        [adid_emb, site_id, site_domain, site_category, app_id, app_domain, app_category])
    # 可以在下面接入残差网络
    for i, dnn_hidden_size in enumerate(params_conf.AD_TOWER_DNN_HIDDEN_SIZES):
        ad_stack = tf.keras.layers.Dense(dnn_hidden_size, activation="relu", name="ad_dense_%s" % i)(ad_stack)
    # 最后一层加不加激活函数？？ 不加可以确保embeding里含有负数
    ad_vector = tf.keras.layers.Dense(params_conf.USER_AD_EMBEDING_SIZE, activation=None, name="ad_vector")(
        ad_stack)

    # 用户塔
    device_id_emb = emb_map["device_id"]
    device_ip_emb = emb_map["device_ip"]
    device_type_emb = emb_map["device_type"]
    device_conn_type_emb = emb_map["device_conn_type"]
    user_input = [inputs_map["device_id"], inputs_map["device_ip"], inputs_map["device_type"],
                  inputs_map["device_conn_type"]]

    user_stack = tf.keras.layers.concatenate(
        [device_id_emb, device_ip_emb, device_type_emb, device_conn_type_emb])
    # 可以在下面接入残差网络
    for i, dnn_hidden_size in enumerate(params_conf.USER_TOWER_DNN_HIDDEN_SIZES):
        user_stack = tf.keras.layers.Dense(dnn_hidden_size, activation="relu", name="user_dense_%s" % i)(user_stack)
    # 最后一层不加激活函数,确保embeding里含有负数
    user_vector = tf.keras.layers.Dense(params_conf.USER_AD_EMBEDING_SIZE, activation=None,
                                        name="user_vector")(user_stack)

    # 添加正则很有必要,否则会报下面的错误
    # loss出错 [predictions must be ＞= 0] [Condition x ＞= y did not hold element-wise:]
    ad_embedding_normalized = tf.nn.l2_normalize(ad_vector, axis=1)
    user_embedding_normalized = tf.nn.l2_normalize(user_vector, axis=1)

    cosine_similarity = tf.reduce_sum(tf.multiply(ad_embedding_normalized, user_embedding_normalized), axis=1,
                                      keepdims=True)

    # 将 cosine 相似度从 [-1, 1] 区间转换为 [0, 1], 值越大表明 user vector 与 item vector 越相似
    cosine_similarity = (cosine_similarity + 1.0) / 2.0

    # 将 cosine 相似度转化为 logits (用于后续计算交叉熵 loss)
    logits = -tf.math.log(1.0 / cosine_similarity - 1.0)
    final_model_outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="final_sigmoid")(logits)

    model = tf.keras.Model(
        inputs=ad_input + user_input,
        outputs=final_model_outputs,
        name="combined_model")

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

    emb_map, feature_inputs_map = build_embeding()
    log_util.info("build embeding finish...")

    # 构建模型
    model = build_model(emb_map, feature_inputs_map)
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


def export_item_embeding():
    output_root_dir = "{}/{}/{}".format(params_conf.BASE_DIR, args.model_output_dir, args.cur_date)
    model_full_input_dir = os.path.join(output_root_dir, "model_savedmodel")

    out_embeding_dir = "{}/{}/{}.txt".format(params_conf.BASE_DIR, args.item_embeding_dir, args.cur_date)

    tf.keras.backend.clear_session()
    log_util.info("start export_item_embeding...")
    # 此处把测试数据item对应的embeding导出,测试数据一般是最近的一天的数据,符合实际情况
    # 多跑一天,把-1带上,防止读数据报错
    # 跑历史多少天的出现的item保存下来embeding,这里灵活选择读取的数据,item相对比较固定,不考虑上下文和用户特征的话,vector接近唯一性
    export_data_date_list = DateHelper.get_date_range(DateHelper.get_date(-1, args.cur_date),
                                                      DateHelper.get_date(0, args.cur_date))
    export_data_date_list.reverse()
    print("train_date_list:" + ",".join(export_data_date_list))

    # load data from tf.data,兼容csv 和 tf_record
    # 仅仅考虑测试集
    _, test_set = data_consumer.get_dataset(args.train_data_dir, export_data_date_list,
                                            get_feature_column_map().values())
    log_util.info("get test data finish ...")

    model = tf.keras.models.load_model(model_full_input_dir)
    log_util.info("load model finish ...")

    # 截取模型的中间部分某层的输出
    item_emb_model = tf.keras.models.Model(
        inputs=model.input,
        # 自己去确定找需要标记de item_id在输入序列中排第几
        outputs=[model.input[0], model.get_layer("ad_vector").output]
    )

    # for features, label in test_set:
    item_embedding = item_emb_model.predict(test_set)
    # print(item_embedding)
    # print(item_embedding[1].shape)
    ad_id_list = item_embedding[0]
    item_vector = item_embedding[1]
    key_set = set()
    with codecs.open(out_embeding_dir, "a", "utf-8") as result_file:
        for (key, value) in zip(ad_id_list, item_vector):
            ad_id = key[0].decode('utf-8')
            embedding_str = ",".join([str(v) for v in value.flatten()])
            if ad_id not in key_set:
                # 根据唯一性id 去重复
                key_set.add(ad_id)
                try:
                    result_file.write("%s\t%s\n" % (ad_id, embedding_str))
                except Exception as e:
                    log_util.error("write embeding error : %s" % e)
                    traceback.print_exc()
                    continue

    log_util.info("write embeding file finish ...")


if __name__ == "__main__":
    print(tf.__version__)
    # run tensorboard:
    # tensorboard --port=8008 --host=localhost --logdir=../log
    args = init_args()
    if args.mode == "train":
        train()
    elif args.mode == "export_item_embeding":
        export_item_embeding()
        
```


码字不易，觉得有收获就点赞、分享、再看三连吧~

欢迎扫码关注作者的公众号： 算法全栈之路

微信公众号: AiStackAll
![](https://gitee.com/ldh521/picgo/raw/master/2021-7-18/1626539300022-qrcode_for_gh_63df84028db0_258.jpg)

知识星球: 搜索 算法全栈之路
![](https://gitee.com/ldh521/picgo/raw/master/img/zsxq_fx.jpeg)