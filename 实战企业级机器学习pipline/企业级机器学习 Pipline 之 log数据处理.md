企业级机器学习 Pipline 之 log数据处理

---

大家都知道，我们现在常用的机器学习平台包括 离线训练 和 在线预估 2 个模块。

其中，离线部分一般负责 **log数据整理**，**样本处理**，**特征处理**和**模型训练**等。
在线部分则包括线上的进行  **实时predict** 过程  (online predict，也称为在线模型的Inference)。

对于一个相对成熟的系统，我们会在 **前端页面** (Html ,App等)通过SDK埋点的方式采集用户的行为日志，一般包括用户各种行为像用户开屏，页面请求、曝光，点击，下载，播放，充值等各种行为，同时也会记录 **服务端** 返回数据的日志，例如：对于一个广告系统来说，就有用户的请求日志和 广告下发日志等，其中，一条请求日志可能对应着多条下发日志记录。

这种系统日志通过flume，kafka ,storm 等大数据处理框架处理之后，会以 hive 表或则 hdfs 文本文件的形式保存在大数据平台上面。一般每条日志保存为log文件中的一行，涵盖可以唯一确定这次用户行为的若干字段，例如 时间戳（timestamp）、androidID、imei、userId、requestid、用户访问页面id 、用户行为等。基本上就是为了区分在某个时间某设备上谁干了什么事这样一个逻辑。
![](https://img.soogif.com/EVjxmeCWbIg9tnL1MzcBCdIEMmgyjKhZ.gif?imageMogr2/thumbnail/!86.29789761832792p&scope=mdnice)

一般我们会使用 hive SQL、 Spark 或 Flink等工具对这些日志进行处理。对于存在hive表中的数据，我们可以有多种方式来读取数据进行处理。
这里主要介绍3中处理方法：

1. hive sql
2. sparksession sql
3. spark rdd

---

### 方法 1， log数据处理之-shell + hive sql
使用 shell 脚本驱动hive sql的方式，来执行sql语句查找源数据若干字段写入固定的hive表。代码示例如下：

```shell

@欢迎关注微信公众号：算法全栈之路
@ filename format_log.sh

#!/bin/bash
source ~/.bashrc
set -x

cur_day=$1
source_table_name=user_xxx_rcv_log
des_table_name=user_xxx_rcv_data
des_table_location="hdfs:/user/base_table/${des_table_name}"

# 如果表不存在则新建表
${HIVE_HOME}/bin/hive  -e "
CREATE EXTERNAL TABLE IF NOT EXISTS ${des_table_name}(${column_name}) PARTITIONED BY (day STRING) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' STORED AS TEXTFILE  LOCATION '${des_table_location}';
ALTER TABLE ${des_table_name} SET SERDEPROPERTIES('serialization.null.format' = '');
ALTER TABLE ${des_table_name} drop partition(day = ${cur_day});
"

# 删除目的地表已有分区
${HADOOP_HOME}/bin/hadoop fs -rm -r -skipTrash ${des_table_location}/day=${cur_day}

$HIVE_HOME/bin/hive  -e " ALTER TABLE ${table_name} drop partition(day = ${cur_day});
ALTER TABLE ${table_name} add partition(day = ${cur_day});
"


# 执行hive sql 写入数据到目的表
${HIVE_HOME}/bin/hive  -e "

set hive.exec.reducers.max = 100;
set hive.execution.engine=mr;
set mapreduce.map.java.opts=-Xmx4096M;
set mapreduce.map.memory.mb=4096;
set mapred.reduce.child.java.opts=-Xmx6g
set mapred.child.java.opts=-Xmx4g
set hive.exec.reducers.max = 100;

insert overwrite table ${des_table_name} partition(day = ${cur_day})
select timestamp,imei,userid,event_tyep from ${source_table_name} where date = ${cur_day}
"

RES=$?
if [ $RES -eq 0 ]
then
	echo "hive job finished!"
	${HADOOP_HOME}/bin/hadoop fs -touchz ${table_location}/day=${cur_day}/_SUCCESS
	exit 0
else
        echo "hive job Error !!!"
	exit -1
fi
```

执行上述shell脚本，可以使用 
```shell
nohup sh -x format_log.sh 20210701 > rcv.log 2>&1 & 
```

---
### 方法 2，log数据处理之-sparksession sql
所谓saprk sql ，就是使用spark session 执行sql语句的方式，来完成数据处理，并把数据保存到hdfs的文本文件上。

<p style="color:red;font-size:18px" >talk is cheap, show the code !!! </p>

这里使用 shell脚本 提交 scala spark 任务的方式进行处理。
scala spark 代码如下：


```scala
@ 欢迎关注微信公众号：算法全栈之路
@ filename format_log.scala

package Data

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

object LogMiddleDataGenerate {
  def main(args: Array[String]) {
    val Array(event_day,all_logdata_path) = args
    val sparkConf = new SparkConf()
    val sparkSession = SparkSession.builder()
    .appName("LogMiddleDataGenerate")
    .config(sparkConf)
    .config("spark.kryoserializer.buffer.max", "1024m")
     .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")      .getOrCreate()
   val sc = sparkSession.sparkContext
   val day = event_day
    // sql 语句
    val all_log_sql = " select timestamp,imei,userid,event_tyep from ${source_table_name} where date  '"+ day +"'"
    val all_log_df = sparkSession.sql(all_log_sql).distinct()
      .rdd
      .map(e=>{
        e.mkString("\t")
      }).persist(StorageLevel.MEMORY_AND_DISK)
 val outputPartNum = math.ceil(all_log_df.count() / 400000).toInt
    all_log_df.repartition(outputPartNum).saveAsTextFile(all_logdata_path)
all_log_df.unpersist()
  }
}
```
执行上诉scala spark任务，需要把上面代码打成jar 包，使用下列代码进行任务提交spark 任务。

```shell
@ 欢迎关注微信公众号：算法全栈之路
@ filename spark_log_format.sh

#!/bin/sh
source ~/.bash_profile

set -x

mvn clean package -U || exit
echo "current working dir: $(pwd)"

day=`date -d "2 day ago" +%Y%m%d`
[ $# -ge 1 ] && day=$1

all_logdata_path=hdfs://dependy_data/all_logdata/${day}

JAR_PATH=./target/fclpc-1.0-SNAPSHOT.jar
class=Data.LogMiddleDataGenerate

${SPARK23_HOME}/bin/spark-submit \
    --master yarn  \
    --deploy-mode cluster \
    --class ${class}  \
    --driver-memory 10G \
    --executor-memory 6G  \
    --conf spark.driver.maxResultSize=8G \
    --conf spark.yarn.priority=VERY_HIGH \
    --conf spark.sql.hive.convertMetastoreParquet=false\
    --conf spark.sql.hive.convertMetastoreOrc=false\
    --conf spark.sql.hive.metastorePartitionPruning=false \
    ${JAR_PATH} \
    ${day}\
    ${all_logdata_path}\
```
执行上述shell脚本，可以使用

```shell
nohup sh -x spark_log_format.sh 20210701 > spark.log 2>&1 & 
```

---
### 方法 3， log数据处理之-spark Rdd
使用spark rdd 和上面使用sparksession sql 差别不是很大，只是使用的sparkcontext接口，直接读取存在 Hdfs 集群上的文件，使用sc.textFile()接口读取文件。和上面差别不大，这里就不在进性详细介绍了。

---
到这里，整个原始数据处理流程就介绍完了，可以使用上面的方法进行符合自己业务的代码改写，觉得有用就点赞和分享吧~

欢迎扫码关注作者的公众号： 算法全栈之路

![](https://gitee.com/ldh521/picgo/raw/master/sfqzzl.jpg)
































