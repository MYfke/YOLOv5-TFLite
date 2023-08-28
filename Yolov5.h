
#ifndef YOLOFACE_YOLOV5_H
#define YOLOFACE_YOLOV5_H


#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

#include "log.h"

// 结构体定义：存储单个检测结果和关键点信息
struct YoloDetectionResult {
    // BoundingBox
    float cx, cy, w, h;                 // 边界框坐标，中心点,宽和高
    float x1, y1, x2, y2;               // 边界框坐标,左上角和右下角
    float score;                        // 置信度
    int label;                          // 标签
    std::string label_text;             // 标签描述
    bool flag;                          // 标记检测是否有效
};


class Yolov5 {
public:
    // 初始化 TFLite
    Yolov5(const std::string &modelFile, const std::string &labelFile, TfLiteType input_type);

    // 推断
    std::vector<YoloDetectionResult> infer(const cv::Mat &ori_image);

    // 释放 TFLite
    ~Yolov5();

    int image_width = 640;
    int image_height = 640;
    int image_channels = 3;

    std::string modelFile;
    std::string labelFile;

    float obj_thres = 0.25; //越低目标越多
    float cls_thres = 0.2; //越低目标越多
    float iou_thres = 0.45; //越高目标越多

    int number_of_threads = 0;

    std::unordered_map<int, std::string> labels;
    TfLiteType input_type;

private:

    static std::unordered_map<int, std::string> readLabels(const std::string &filename);

    static std::vector<uchar> matToVectorRGB(const cv::Mat &image);

    static std::vector<float> normalize(const std::vector<uchar> &vec);

    template<typename T>
    std::vector<T> forward(std::vector<T> inputData);

    std::vector<YoloDetectionResult> nms(const std::vector<float> &outputdata);

    // tflite 相关变量
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;


    unsigned long int max_result = ULONG_MAX;
    bool profiling = false;
    bool allow_fp16 = true;
    bool gl_backend = false;

    bool NNAPI_delegate = true;
    bool hexagon_delegate = false;
    bool xnnpack_delegate = false;

    float input_mean = 127.5f;
    float input_std = 127.5f;


    template<typename T>
    std::vector<T> quantize(const std::vector<float> &input, float scale, T zero_point);

    template<typename T>
    std::vector<float> dequantize(const std::vector<T> &input, float scale, T zero_point);

};


#endif //YOLOFACE_YOLOV5_H
