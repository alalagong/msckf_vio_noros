# msckf_vio_no_ros

#### 介绍
该代码源地址为：https://github.com/KumarRobotics/msckf_vio/
在原代码的基础上该代码去除了对ROS和boost库的依赖，仅用于学习交流

#### 安装教程
该工程提供了premake(premake5.lua)和cmake(CmakeLists.txt)两种方式编译：

Windows：

third_libraries目录下已经包含OpenCV3.1(支持vs2013及以后版本)和Eigen3.3.7
方式1使用premake编译：
1. 修改scripts目录下premake_config.bat脚本，并修改visual studio版本
2. 双击premake_config.bat即可生成visual studio工程

方式2使用cmake编译：
运行scripts目录下cmake_configure.bat脚本，如遇到问题建议自行修改CMakeLists.txt文件和对应的脚本

Linux：
premake在Linux下未测试，建议使用cmake来编译，编译过程略
	
注：代码可通过USING_VIZ宏定义来确定是否使用OpenCV的VIZ模块来显示结果。
premake5.lua和CmakeLists.txt默认设置为false

#### 使用说明
可参照原代码的说明下载公开数据集，其中EuRoC数据集地址为：
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
由于msckf_vio需要静止状态下初始化，建议下载初始状态为静止状态的数据集，如：V1_01_easy

注意：
1. 默认使用entry/config/camchain-imucam-euroc.yaml中的相机外参，若使用其它数据集需要对应修改
2. 通过entry/main.cpp修改数据集路径或通过命令行参数输入数据集路径


