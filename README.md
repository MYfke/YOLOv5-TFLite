# YOLOv5 TFLite 部署

本 demo 实现了使用 TensorFlow Lite (TFLite) 部署 YOLOv5 模型。

使用 C++ 接口加载量化后的 YOLOv5 模型，并在图像上进行目标检测。
这个 demo 演示了在 windows 平台运行结果。以及可以直接集成进嵌入式设备和 Android 设备。

## 前提条件

在开始之前，请确保满足以下要求：

- 安装 MSVC 2019 工具链（不要使用 MinGW 工具栏）
- 下载 YOLOv5 模型的权重文件（.tflite 文件，本项目已集成）
- 链接 OpenCV 库（本项目已集成）
- 链接 TensorFlow Lite 库（本项目已集成）
- 本项目暂仅支持 fp32、fp16、int8、uint8 精度模型

## 使用方法

**直接编译运行就ok**

运行后，程序将加载图像，执行目标检测并显示结果。

## 注意事项

### Windows x64 库的编译环境

- 本项目使用 TensorFlow Lite 版本 2.7.4。
- 在 Windows x64 平台上，tensorflowlite.dll 采用 MSVC 2019 工具链编译，因此运行时最好采用同样的工具链。
- include 文件夹下你只需要使用头文件，源码及其他文件可以删除。
- flatbuffers 是 TFLite 的外部依赖库。
- OpenCV 4.8.0 已集成在项目中，部署时可更换为主机上的 OpenCV。

_tensorflowlite 编出来的库，只能在对应编译版本运行。_
_Release 库必须在 Release 下运行，Debug 库必须在 Debug 下运行。_
_**这是玄学**_ 

### ARMv8 库的编译环境

- 采用 Bazel 生成编译配置，版本为 4.2.3。
- 使用 NDK 版本 21e (21.4.7075529)，目标 API 30。
- 编译目标为 Android SDK API 30。
- Android SDK Build Tools 采用 30.0.3。
- flatbuffers.lib 直接克隆仓库自行编译。


