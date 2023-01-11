docker 快速上手使用指南

作为一个对技术有追求的程序员，购买了一个自己的阿里云机器，但中途历经多次数据和环境的迁移，发现多次配置阿里云环境太费事了，最终下定决心把一些基础服务 docker 化。查了一些资料，也总结了一些常用的docker 工具的配置方法，这里和大家分享一下，希望能对大家有所帮助，有问题欢迎 留言讨论～

Docker是基于Go语言进行开发实现的一个开源的应用容器引擎，每个容器内运行着一个应用，不同的容器相互隔离，容器之间也可以通过网络互相通信。docker 作为一种轻量级的沙盒，以其一次部署多地运行和管理简单、快速方便部署等优点，可以很方便的解决业务部署上的痛点问题而大受欢迎。对于docker的原理这里不在展开，本文仅仅从docker的快速上手使用方面展开阐述。
![](https://img.soogif.com/jt7vsvuXEihTNosROunNy9WdwHX5qeDY.gif?imageMogr2/thumbnail/!17.699376510349406p&scope=mdnice)


---

### docker 的安装与基础命令

#### (1.1) docker 安装
windows下使用docker的话，需要下载docker for windows 的安装文件，安装后将bin文件配置到环境变量中，就可以通过cmd使用命令行对容器进行操作和管理了，当然windows下也可以通过桌面版UI进行管理。

因为docker大多数在linux服务端使用，这里仅仅详细介绍linux/ubuntu 下的安装过程：
安装命令(使用阿里云镜像)：

`sudo apt-get update && sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common && curl -fsSL https:``//download``.[docker.com](http://docker.com/)``/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository "deb [arch=amd64] [http://mirrors.aliyun.com/docker-ce/linux/ubuntu](http://mirrors.aliyun.com/docker-ce/linux/ubuntu) $(lsb_release -cs) stable" && sudo apt-get update && sudo apt-get install -y docker-ce`

一条命令就可以搞定安装，不过安装可能需要一些时间，等着就行。


在使用过程中，当需要查找某个镜像的时候，我们可以去docker中央仓库：https://hub.docker.com/  查看，从中我们可以看到该名称镜像历史的release的版本，我们可以根据需求选择自己需要的版本。


---


 #### (1.2) docker常用管理命令
(1) 拉取某个镜像 

`docker pull  xxxx  `

(2)查看镜像与删除镜像

查看所有镜像：`docker images `

删除镜像 ： `docker rmi xxxx `

 注意：删除镜像前必须先停用、删除该镜像产生的实例
 
(3) 查看与删除实例

查看所有实例： `docker ps -a `

查看activate实例： `docker ps `

删除实例： `docker rm xxx`

(4)根据镜像创造docker 实例,这里以nginx为例

`docker run --name nginx  -p 80:80 -v /root/docker_conf/nginx/nginx.conf:/etc/nginx/nginx.conf  -d docker.io/nginx `

其中: -name是镜像的名称,  -p 是端口映射, -v 是绑定一个卷, 我们可以把自己的文件目录映射到镜像内部目录, 冒号前面为自己的实际机器, :后面是对应镜像的目录。 

(5) 当docker 启动失败的时候debug原因：

`docker logs  xxID `

(6) 进入实例 

根据上面的命令创建的实例之后，如果成功的话会默认启动实例，我们可以使用下列命令进入到该实例内部：

 `docker exec -it nginx  bash`
 
(7) 复制文件到实例内部系统

`docker cp my.cnf mysql5.7:/etc/mysql/my.cnf`


(8)  实例内系统安装软件

实例内也是一个操作系统，我们可以安装需要使用的程序：

`apt update `

 例如安装vim命令 :  `apt install vim `
 
(9) 停止和启动docker 实例

`docker start xxx`

`docker stop xxx `

(10)  卸载docker

`yum remove docker-ce`

删除镜像、容器、配置文件等内容：

`rm -rf /var/lib/docker `

![](https://img.soogif.com/KPjORwFreUV0UBTeRZi3mnV3PWw0mA4w.gif?scope=mdnice)

---

### (二) docker 资源管理命令

docker 镜像在很多时候会耗占比较多的内存和cpu, 甚至某个镜像被用来挖矿的情况下, cpu会达到100%(都是血与泪的实践经历啊)， 所以我们可以使用下面的命令来限制性设置某个实例的内存与cpu占用量。


(1) docker 各个镜像资源使用情况查看(包括cpu和内存)：

`docker stats `

(2)  限制docker 内存

`docker update -m 20M --memory-reservation 20M --memory-swap 20M nginx`

（3）限制docker cpu 

`docker update  --cpu-period=100000 --cpu-quota=20000 nginx`

在每 100 毫秒的时间里，运行进程使用的 CPU 时间最多为 20 毫秒，这里仅仅考虑百分比。

---

### (三) docker 常用软件管理

在这里，作者分享一些自己常用的一些docker 软件的配置方法 ，包括wiz note , mysql, redis, nginx, gitea , graph learn 等。以后会不断的更新，在使用过程中，如果遇到任何问题，欢迎去： 算法全栈之路 公众号 和作者讨论 ～ 

---

#### (3.1) docker 部署私有的笔记服务 wiznote 

作为一个对写代码有追求的程序员，有一个自己用起来顺手的笔记服务是必不可少的，作者经过大量的筛选，目前主要在使用的就是wiznote 这个软件，个人用户使用5个账号内免费，并且个人功能上的需求完全可以满足，使用docker部署也非常简单便利。

(1) 部署过程如下：

`docker run --name wiz --restart=always -it -d 
-v  /root/docker_conf/wiz_wiki_dat:/wiz/storage  
-v  /etc/localtime:/etc/localtime 
-p 9191:80 -p 9269:9269/udp 
wiznote/wizserve`

然后我们访问 `http://ip:9191` 网址即可以访问我们自己的笔记后端服务了。

初始账号： `admin@wiz.cn`,  密码：`123456，`
输入后记得自行进行修改啊，否则可能导致被他人登陆。


---

#### (3.2) docker 部署 mysql

（1）docker 部署 mysql 

参考wiki: https://blog.csdn.net/weixin_43888891/article/details/122518719

```
docker pull mysql:5.7
docker run -itd --name mysql5.7 --restart=always -p 3306:3306 -e MYSQL_ROOT_PASSWORD=you_mysql_pwd docker.io/mysql:5.7 
进入docker 内部： docker exec -it mysql5.7 bash
```


(2) 配置数据库可以远程访问

```
docker exec -it mysql5.7 bash
grant all on *.* to root@'%' identified by 'you_mysql_pwd' with grant option;

```

(3) 优化docker MySQL 配置

更新 my.cnf 文件


```
[mysqld]

max_connect_errors = 1000

lower_case_table_names = 1

performance_schema_max_table_instances = 200

table_definition_cache = 100

table_open_cache = 100

innodb_buffer_pool_size=2M

performance_schema=off
```


更新 docker.cnf  文件 

```
skip-host-cache
skip-name-resolve 
```


（4）覆盖docker 内部文件配置

```
docker cp docker.cnf  mysql5.7:/etc/mysql/conf.d/docker.cnf
docker cp my.cnf  mysql5.7:/etc/mysql/my.cnf
```

(5)  限制docker MYSQL 内存和cpu使用

```
docker update -m 800M --memory-reservation 800M --memory-swap 800M mysql5.7
docker update  --cpu-period=100000 --cpu-quota=20000 mysql5.7 
```


---

#### (3.3)docker 部署 redis 

(1) docker 使用redis
```
docker run --name redis -p 6379:6379 -v /root/docker_conf/redis/redis.conf:/etc/redis/redis.conf -d redis redis-server /etc/redis/redis.conf
```

我们可以把redis常用的配置文件放在 /root/docker_conf/redis/redis.conf 这个路径下。

(2) 外面访问容器里的redis服务 

`docker exec -it redis redis-cli`

(3) 限制redis内存访问 

`docker update -m 50M --memory-reservation 50M --memory-swap 50M redis `

然后我们就可以通过 ip: 6379 端口访问我们的redis 服务了。

---

#### (3.4)  docker 部署 nginx 

(1) docker 使用 nginx

`docker pull nginx `

// 本机卷映射 nginx卷

`docker run --name nginx -p 80:80 -v /root/docker_conf/nginx/nginx.conf:/etc/nginx/nginx.conf -d docker.io/nginx`

(2) 限制docker 使用内存 

`docker update -m 50M --memory-reservation 50M --memory-swap 50M nginx`

注意，使用docker 之后，映射的服务IP应该改成外部IP，在使用127.0.0.1会报错。

（3）配置nginx conf 

`/root/docker_conf/nginx下nginx.conf`

这样我们修改本机器上的/root/docker_conf/nginx/nginx.conf  路径下的配置文件，就可以直接影响docker nginx服务的配置了。

然后我们就可以通过 http://ip:80  端口访问我们的 nginx  服务了。

---

#### (3.5) docker 部署 gitea 

在很多时候我们会需要部署我们自己的git版本管理仓库，可以使用gitea 这个开源版本库，非常好用，使用docker部署也非常简单。

(1) 拉取gitea仓库 

`docker pull gitea/gitea`

(2)  启动gitea 实例与服务

`docker run -d --privileged=true --restart=always --name=gitea -p 20022:22 -p 3000:3000 -v /root/docker_conf/gitea:/data gitea/gitea:latest`

在下面我们就可以使用 `http://ip:3000 ` 去访问我们的git服务了。

---


#### (3.6)  docker 部署使用图机器学习框架graph learn  


注意python版本：python2.7 

(1) 安装软件

`pip2 install tensorflow==1.13.1`

(2) 拉取 graph learn镜像

`sudo docker pull graphlearn/graphlearn:1.0.0-tensorflow1.13.0rc1-cpu`

(3) 运行 graph learn 实例

`sudo docker run  -d -it  --name graph_idml -v  /home/local_machine/workspace:/root/workspace  graphlearn/graph-learn` 

(4) 进入graph leanr 实例

`sudo docker exec -it 6c1c2dda75f9 /bin/bash`

(5) 开始使用 graph learn 框架训练图机器学习模型

在这里，我们把训练模型需要的数据下载到 /home/local_machine/workspace  这个路径下，然后进入到镜像里面，使用镜像的graph learn 环境，训练自己的模型，输入数据和执行的代码选择  /root/workspace 这个路径。

`python train_unsupervised.py`
![](https://img.soogif.com/vgz6OZ2UXgaW816kriaordOnfnWl3bMW.gif?scope=mdnice)


---

### (四) docker 镜像文件保存与加载

(1) 保存docker 镜像文件到本地：

 `docker save  nginx  >./nginx.tar  `
 
(2)  在另一台主机加载本地文件到镜像 ： 

`docker load < nginx.tar`

然后看 docker images ，就能看到该镜像 ，但是名字和标签都是none

（3）执行完上述语句后，查看本地镜像，会看到新加载的镜像名字和标签都是none，利用该镜像的id对名字和标签重新命名即可：

```
指令：docker tag 镜像id 镜像名:标签

docker tag 172825a55619 confluence6.12_cracked:0.1
```

(4) 启动镜像 

`docker run -it --name nginx 172825a55619 /bin/bash`

---

### (五)  docker 镜像文件的提交与上传到中央仓库 

很多时候，我们会需要在已有的镜像上进行一些自定义修改，然后重新打包上传自己的镜像供别人使用，我们可以使用下面的流程来上传自己新打包的镜像到中央仓库。

(5.1)  官网账号注册  

   首先我们先到docker官网注册一个账号，这样我们才能将制作好的镜像上传到docker仓库，
   
  打开 `https://hub.docker.com/ `
  
(5.2) 提交自己要上传的镜像 

 `docker commit Container_name yourdockerhub/nginx:latest`
 
注意：commit 对包名命名有要求，Container_name 容器名称,yourdockerhub改成自己的账号，否则无法上传

(5.3)  登录到远程docker仓库(输入自己注册的账号和密码)

  `docker login`
  
(5.4)  上传至docker云端

`docker push yourdockerhub/nginx:latest `

 注意: 这里push 对包名命名有要求，yourdockerhub改成自己的账号，否则无法上传 
上传成功了就可以在中央仓库你的账号下搜到你刚上传的镜像了。


到这里，docker 快速上手使用指南已经结束了。

---


码字不易，觉得有收获就点赞、分享、再看三连吧~

欢迎扫码关注作者的公众号： 算法全栈之路 

![](https://gitee.com/ldh521/picgo/raw/master/2021-7-18/1626539300022-qrcode_for_gh_63df84028db0_258.jpg)


