基于springboot +  Semantic UI 的个人博客系统（源码+开发过程的详细视频+部署说明，超详细，非常推荐！！！）

---

今天在这里给大家推荐一个博客系统，可以自定义修改界面和功能哦 ！界面和功能说明如下：

**[算法混子博客](http://blog.zhihang.info/ "算法混子博客")**


***
###### 界面如下 ：
（1）首页

![](https://gitee.com/ldh521/picgo/raw/master/blog.jpg)

（2）后台管理页面

![](https://gitee.com/ldh521/picgo/raw/master/blog_admin2.jpg)

部署说明：
**源码获取途径：**

（1）
[程序员资源网](http://it.zhihang.info/article/detail/109.html "程序员资源网")
http://it.zhihang.info/article/detail/109.html

（2）关注公众号  算法全栈之路

回复关键字： 算法混子博客  获得源码

---

**作者开发视频说明**

[SpringBoot开发一个小而美的个人博客](https://www.bilibili.com/video/BV1Pt4y1U7hv?from=search&seid=793468435859227137 "SpringBoot开发一个小而美的个人博客")

https://www.bilibili.com/video/BV1nE411r7TF?from=search&seid=793468435859227137


**部署说明：**

（2.1)下载源码，解压 zip, 用 idea 打开工程文件

 (2.2) 因为工程数据库是使用Hibernate接口构建的，所以工程只需要在mysql数据库里构建工程使用的库就可以，表会自动构建。具体修改见resources/application.yml 和 application-xxx.yml.

可以在数据库user表插入一条记录，登录管理后台

http://localhost:8080/admin


```sql
管理员	admin	2ea743e61910f6fa16e4de2d37cdf308	240238650@qq.com	tou.jpg	男	1460	1	4	0	管理员	2021-07-23 16:38:48	2021-08-28 19:09:20
```

 (2.3) 在服务上部署. 
 (i)把工程用mvn clean package 打jar包。
 (ii) 用springboot 特有的jar包部署，springboot 内嵌带有tomcat，所以直接部署就行。使用以下命令：
```shell
@ 欢迎关注公众号  算法全栈之路

nohup java -jar ./code-0.0.1-SNAPSHOT.jar --spring.profiles.active=prod > core.log 2>&1 &
```

在浏览器中访问：

http://localhost:8080




---
欢迎扫码关注作者的公众号： 算法全栈之路

![](https://gitee.com/ldh521/picgo/raw/master/sfqzzl.jpg)


