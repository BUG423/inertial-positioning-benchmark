# FAST-LIVO 环境配置指南

> 来源：原 `fast-livo-setup` 仓库，经规范化后并入本 benchmark。验证环境为 Ubuntu 20.04 + ROS Noetic。\n\n> [!CAUTION]\n> 本文记录的是 ROS 1 时代的历史兼容环境。依赖版本、上游提交和系统支持状态可能变化；执行安装前应核对各上游项目的当前文档，不要将本文视为自动化安装脚本。

## 1. Ubuntu 与 ROS

| 项目 | 说明 |
|------|------|
| **推荐版本** | Ubuntu 16.04 ~ 20.04 |
| **原因** | 官方 ROS 1 最高支持到 Ubuntu 20.04（之后为 ROS 2） |
| **安装类型** | `ros-*-desktop-full`（避免后续手动补装功能包） |

本指南使用的环境：**Ubuntu 20.04 + ROS Noetic**。

> 如需安装额外的功能包：`sudo apt install ros-noetic-<PACKAGE>`

## 2. PCL（点云库）

直接通过 apt 安装即可，**不推荐源码编译**（耗时且易出错）：

```bash
sudo apt install libpcl-dev
```

## 3. Eigen

必须使用 **Eigen 3.4.0**，其他版本可能存在兼容性问题。

1. 从 [Eigen 官网](https://eigen.tuxfamily.org/index.php?title=Main_Page) 下载 `Eigen 3.4.0` 源码
2. 解压后编译安装：

```bash
mkdir build && cd build
cmake ..
make
sudo make install
```

> ⚠️ **不要使用 apt 安装的 Eigen**，也不要使用 3.4.0 以外的版本。

## 4. OpenCV

直接通过 apt 安装，**不推荐源码编译**：

```bash
sudo apt install libopencv-dev python3-opencv
```

### 版本兼容性注意事项

- OpenCV **3.x** 使用 `CV_<CONSTANT>` 形式的宏（如 `CV_8UC1`）
- OpenCV **4.x** 改用 `cv::<constant>` 命名空间形式

如果编译时因为版本差异出现相关报错，全局查找替换即可。

## 5. Sophus（非模板类版本）

FAST-LIVO 依赖的是 **非模板类** 的 Sophus，需要手动编译并修复一处源码问题。

### 5.1 克隆并切换分支

```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
```

### 5.2 修复 `so2.cpp`

打开 `Sophus/sophus/so2.cpp`，将以下代码：

```cpp
SO2::SO2()
{
  unit_complex_.real() = 1.;
  unit_complex_.imag() = 0.;
}
```

替换为：

```cpp
SO2::SO2()
{
  // unit_complex_.real() = 1.;
  unit_complex_.real(1.);
  // unit_complex_.imag() = 0.;
  unit_complex_.imag(0.);
}
```

> **原因**：`unit_complex_` 实例的 `real()` 和 `imag()` 方法返回的是值，不能通过 `=` 直接赋值。

### 5.3 编译安装

```bash
mkdir build && cd build
cmake ..
make
sudo make install
```

## 6. Vikit（视觉惯性工具包）

```bash
cd ~/catkin_ws/src          # 如果没有该目录则自行创建
git clone https://github.com/uzh-rpg/rpg_vikit.git
```

## 7. Livox ROS Driver

需要先编译 Livox SDK，再安装 ROS 驱动。

### 7.1 编译 Livox SDK

```bash
git clone https://github.com/Livox-SDK/Livox-SDK.git
cd Livox-SDK
cd build && cmake ..
make
sudo make install
```

### 7.2 安装 livox_ros_driver

```bash
git clone https://github.com/Livox-SDK/livox_ros_driver.git ~/ws_livox/src
cd ~/ws_livox
catkin_make
echo "source ~/ws_livox/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 8. 编译 FAST-LIVO

所有依赖安装完毕后，编译 FAST-LIVO 本身：

```bash
cd ~/catkin_ws/src
git clone https://github.com/hku-mars/FAST-LIVO.git
cd ..
catkin_make
source ~/catkin_ws/devel/setup.bash
```

> ✅ **只要各依赖版本正确，基本不会遇到大问题，功德圆满。**

## 9. 测试运行

编译完成后，用官方提供的 bag 文件验证：

```bash
# 终端 1：启动 FAST-LIVO
roslaunch fast_livo mapping_avia.launch

# 终端 2：播放测试数据
rosbag play YOUR_DOWNLOADED.bag
```

## 来源与维护\n\n本文整理自 FAST-LIVO 相关上游文档与实际复现记录。新增命令或版本约束时，应同时注明验证日期、操作系统、ROS 版本和对应上游提交。
