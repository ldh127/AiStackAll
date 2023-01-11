算法工程师打死都要记住的20条常用shell命令

---

**1. 使用 hadoop 命令查看 dfs上的文件,"\t"分隔**

```shell
hadoop fs -text hdfs/user/app/data.20210701/* | awk -F '\t' '{ print  $1 "\t"  $2  }' | less -10
```
**2. 查看 hdfs 上文件的大小，2种方法**

(2.1) -du 查看大小  -h （human，以人可以看懂的方式展现）
```shell
hadoop fs -du -h /hdfs/user/app/data.20210701
```
（2.2）有的hadoop 不支持 -h 参数，使用以下的方式，单位MB
```shell
hadoop fs -du /hdfs/user/app/data.20210701  |  awk -F' '  '{printf "%.2fMB\t\t%s\n", $1/1024/1024,$2}'
```
**3. shell命令中间添加 if 判断语法**
```shell
hadoop fs -text /hdfs/user/app/data.20210701/* | awk -F'\t' '{if($14 != "-")  print $14}'  | less -10
```
**4. 统计Hadoop 文件中共有多少列**
```shell
hadoop fs -text /hdfs/user/app/data.20210701/* | awk -F "\t" '{print NF}' | less -10
```
**5.统计hadoop上文件行数**
```shell
hadoop fs -text /hdfs/user/app/data.20210701/* | wc -l
```
以上命令使用 cat sample.txt |  同理可以使用。

**6. kill掉集群中的 spark 任务**
```shell
yarn application -kill  application_xxxx
```
**7. kill掉集群中的 MR 任务**
```shell
hadoop job -kill application_xxxx
```
**8. 查找文件中含有某个字符串的行**
```shell
cat sample.txt | grep "str_non" | less -10 
cat sample.txt | grep -C 5 "str_non" | less -10 # 上下5行
```
**9. 遍历文件中各列，以及隔行，进行判断自定义处理方式**
```shell
cat sample.txt | awk -F "\t" 
'{
cnt=0; for(i=1; i<=num; i++)
{if($i!=0){cnt=1; break}}; 
 if(cnt==1)printf("%s\n",$0)
}'
num=15
```
**10. Linux文件编码转换**
```shell
#（10.1）. 通过
   iconv -l 
#命令查看，其支持的编码格式还不少，之间可以互相转换
#（10.2）. 转换gbk编码文件为utf-8编码文件
#简洁命令：
iconv -f gbk -t utf-8 index.html > aautf8.html
#其中-f指的是原始文件编码，-t是输出编码  index.html 是原始文件 aautf8.html是输出结果文件
#（10.3. 转换gbk编码文件为utf-8编码文件详细命令：
iconv -c --verbose  -f gbk -t utf-8 index.html -o index_utf8.html
#-c 指的是从输出中忽略无效的字符， --verbose指的是打印进度信息 -o是输出文件****
```
**11. spark-shell 统计多列覆盖率,该方法非常好用**
```scala
val df=spark.read.textFile("/hdfs/user/app/data.20210701/*").map(e=>(e.split("\t")(4),e.split("\t")(5))).toDF("appname","flag").cache();
val re=df.agg(
   (sum(when($"appname"===("-"),0).otherwise(1))/count("*")).as("appnamec"),
   (sum(when($"flag"===("-"),0).otherwise(1))/count("*")).as("flagC")
 ).show()
```
**12. hadoop 跨集群复制文件**
```shell
hadoop distcp -su source_ugi -du target_ugi source_path target_path
```
**13. 内网跳板机文件复制命令**
```shell
scp -r root@ip:/home/workspace/file.txt  pwd
```
**14. 批量杀死Linux上含有某个参数的命令**
```shell
ps -ef |grep "param" | awk '{print $2}'| xargs kill -9
```
**15. 查看当前目录下各个文件大小，同目录**
```shell
du -h --max-depth=0 *
```
**16. 修改linux上某个目录的权限**
```shell
chown -R  work  /home/workspace
```

**17. 查看CPU 相关参数，c++程序员关注线程绑定**
```shell
# 查看物理CPU个数
cat /proc/cpuinfo|grep "physical id"|sort -u|wc -l
# 查看每个物理CPU中core的个数(即核数)
cat /proc/cpuinfo|grep "cpu cores"|uniq
# 查看逻辑CPU的个数
cat /proc/cpuinfo|grep "processor"|wc -l
# 查看CPU的名称型号
cat /proc/cpuinfo|grep "name"|cut -f2 -d:|uniq
# Linux查看某个进程运行在哪个逻辑CPU上
ps -eo pid,args,psr | grep nginx 
```
**18. Java环境下jar命令可以解压zip文件，神器！！**
```shell
# jar 命令解压zip
jar -xvf appinf.zip .
```
**19. python 可以读取parquet文件**
```python
# 方法1 , pd接口读取
import pandas as pd
pdata=pd.read_parquet('/hdfs/user/app/data.20210701.parquet', engine='fastparquet')
# 方法2 ,使用fastparquet
from fastparquet import ParquetFile
pf =
ParquetFile('/hdfs/user/app/data.20210701.parquet')
df = pf.to_pandas()
print(df[0:2])
```

**20.取得时间的若干脚本**
```shell
num=1
[ $# -ge 1 ] && num=$1
day=`date -d "${num} days ago" +'%Y%m%d'`
beforeDay=`date -d "${day} -1 days" +%Y%m%d`
```

---
到这里，算法工程师常用shell命令详解就介绍完了，觉得有用就点赞和分享吧~

欢迎扫码关注作者的公众号： 算法全栈之路

![](https://gitee.com/ldh521/picgo/raw/master/sfqzzl.jpg)


