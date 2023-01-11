快速搭建免费内网穿透工具 ( 服务端aliyun + 客户端 windows )

---
**服务端aliyun + 客户端 windows 免费内网穿透工具** 公网Ip映射

先介绍下背景，最近在做一个只有用户关注公众号才能在某个网站进行下一步的操作的功能，需要不断的进行微信公众号调试。然而微信公众号仅仅支持公网ip的服务配置服务器地址(URL)，所以就需要一个公网ip 来在本地搭建公众号后台。为方便调试，so 可以试试这个 免费内网穿透工具哦

试了下花生壳和natApp这 2 个工具，发觉natApp的免费版本特别不稳定，而最新版的花生壳需要自己去注册一个公网域名才可以使用。衡量之下，自己就花了点时间找了个 **免费工具(frp)** , 然后自己把测试的可以使用了，就分享出来，不喜勿喷哈 ~

---

#####使用方法

说明：这个工具是 linux(阿里云)服务端 + windows 客户端配置使用。理所当然，一个服务端可以相应对应很多的客户端APP 。

######所需要的软件下载：

linux 服务端:
```shell
wget https://github.91chifun.workers.dev//https://github.com/fatedier/frp/releases/download/v0.34.3/frp_0.34.3_linux_amd64.tar.gz
```
Windows 客户端：
```shell
wget https://github.91chifun.workers.dev//https://github.com/fatedier/frp/releases/download/v0.34.3/frp_0.34.3_windows_amd64.zip
```
工具源网站：https://github.com/fatedier/frp/releases

国内的网络不好的用户，可以使用下面地址下载这个软件使用哦

[程序员资源网](http://it.zhihang.info/article/detail/110.html "程序员资源网")

http://it.zhihang.info/article/detail/110.html

---

######配置过程说明：
（1）服务端安装

```shell
解压： tar -xf frp_0.34.3_linux_amd64.tar.gz
       cd frp_0.34.3_linux_amd64
编辑frps.ini服务端配置：
[common]
bind_port = 7000
vhost_http_port = 6666
注意：该目录下有多个文件，注意看清楚名字 **frps.ini** 表示服务端配置server。
```
启动服务端命令可以使用如下命令：
```shell
nohup ./frps -c ./frps.ini  > x.log 2>&1 &
```

（2）客户端安装

解压过程此处不表

修改客户端配置frpc.ini

```shell
[common]
server_addr = 47.95.196.216  # 上面服务端ip
server_port = 7000  # 服务端端口，这个默认就行不用改

[web]  # 声明是一个web服务
type = http
local_ip = 192.168.31.13  # 本机ipconfig 得到的ipv4,需修改
local_port = 1001    # 本机服务启动的端口号，需修改
remote_port = 6666   # 和上面服务端绑定的端口号
custom_domains = frp.zhihang.info # http服务需要的域名。
```
在Windows端启动client服务可以使用如下命令
```shell
.\frpc.exe -c .\frpc.ini
```

现在使用访问请求 `http://frp.zhihang.info:6666/ `就可以完成 **公网域名** 到 **本机服务** 192.168.31.13:1001的映射啦~，访问本机服务直接使用:`http://frp.zhihang.info:6666/`就ok啦。

当然，fcr也可以配合nginx使用，上面的地方全都不需要修改，只要在nginx.conf里面加上如下配置就可以啦
```shell
server {
        listen       80;
        proxy_set_header  X-Real-IP  $remote_addr; #记录远程访问ip，方便应用中获取
        server_name  frp.zhihang.info;
        location / {
            proxy_pass   http://frp.zhihang.info:6666; #将域名为frp.zhihang.info的请求分发到本地6666端口的服务
        }
    }
```

最后一点，因为我是使用的阿里云服务器作为服务器，所以需要在阿里云里加上相关端口的开启哦。同时，这里的 frp.zhihang.info，也是我申请的 zhihang.info二级域名下的三级域名。

我们申请了一个二级域名，就拥有了大量的三级域名，可以解决很多域名不够用的问题哦，使用的时候只需要在阿里云的dns里进行配置即可。

---

福利：如果你使用内网穿透频率不是很高的话，可以使用我上面的配置不用修改，只安装本地客户端，然后把你的服务端口号设定为你设定的号码,然后就可以使用 `http://frp.zhihang.info`访问你的本机服务了哦。

---

到这里，搭建免费内网穿透工具就已经介绍完成啦，觉得有收获就点赞、分享、再看三连吧~

欢迎扫码关注作者的公众号： 算法全栈之路

![](https://gitee.com/ldh521/picgo/raw/master/sfqzzl.jpg)
