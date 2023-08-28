
#include "Yolov5.h"


Yolov5::Yolov5(const std::string &modelFile, const std::string &labelFile, TfLiteType input_type)
        : modelFile(modelFile), labelFile(labelFile) {

    // 1. 加载模型
    model = tflite::FlatBufferModel::BuildFromFile(modelFile.c_str());
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // 2. 创建解释器
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // 3. 使用 NNAPI 代理 (仅当不在 Windows 上)
#if USE_NNAPI
    if(NNAPI_delegate){
            tflite::StatefulNnApiDelegate nnapi_delegate;  // 创建 NNAPI 代理实例
    TFLITE_MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(&nnapi_delegate) == kTfLiteOk);
    }
#endif

    // 4. 设置输入并调用解释器
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    LOG(INFO) << "======= Pre-invoke Interpreter State =======\n";
    tflite::PrintInterpreterState(interpreter.get());

    interpreter->SetAllowFp16PrecisionForFp32(allow_fp16);  //允许在进行推断时使用半精度浮点数 (FP16) 代替全精度浮点数 (FP32)
    interpreter->SetNumThreads(number_of_threads);  //设置 TensorFlow Lite 推理过程中线程池的大小, 0会选择默认的线程数

    // 打印模型详细信息
    LOG(INFO) << "=============== Model Details ==============\n";
    LOG(INFO) << "tensors size: " << interpreter->tensors_size();   //模型中的张量总数
    LOG(INFO) << "nodes size: " << interpreter->nodes_size();       //操作节点（或层）的数量
    LOG(INFO) << "inputs: " << interpreter->inputs().size();        //输入张量的数量

    // 打印每一个张量信息
    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
        if (interpreter->tensor(i)->name)
            LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                      << interpreter->tensor(i)->bytes << ", "
                      << interpreter->tensor(i)->type << ", "
                      << interpreter->tensor(i)->params.scale << ", "
                      << interpreter->tensor(i)->params.zero_point;
    }

    this->labels = readLabels(labelFile);
    this->input_type = input_type;
}


std::unordered_map<int, std::string> Yolov5::readLabels(const std::string &filename) {
    std::ifstream file(filename);
    std::unordered_map<int, std::string> labels;

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int key;
            char colon;
            std::string value;

            // 解析格式如 "0: person" 的行
            if (iss >> key >> colon >> value) {
                labels[key] = value;
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    return labels;
}


std::vector<uchar> Yolov5::matToVectorRGB(const cv::Mat &image) {
    std::vector<uchar> image_vector;

    if (!image.empty()) {
        int total_elements = image.total() * image.channels();
        image_vector.resize(total_elements);  // Resize the vector to hold the image data

        int index = 0;  // Index for the image_vector

        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
                image_vector[index++] = pixel[2]; // R channel
                image_vector[index++] = pixel[1]; // G channel
                image_vector[index++] = pixel[0]; // B channel
            }
        }
    }

    return image_vector;
}


std::vector<float> Yolov5::normalize(const std::vector<uchar> &vec) {
    std::vector<float> normalized_vec;
    for (float value: vec) {
        value /= 255.0;
        normalized_vec.push_back(value);
    }
    return normalized_vec;
}


template<typename T>
std::vector<T> Yolov5::forward(std::vector<T> inputData) {
    // 记录启动时间
    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();

    // Get the input tensor pointer
    int inputTensorIndex = interpreter->inputs()[0];

    // Get input buffer of type T
    auto *inputBuffer = interpreter->typed_tensor<T>(inputTensorIndex);

    // Copy inputData data to the input tensor
    for (size_t i = 0; i < inputData.size(); i++) {
        inputBuffer[i] = inputData[i];
    }

    // Inference
    interpreter->Invoke();

    // Get Output
    int outputTensorIndex = interpreter->outputs()[0];

    // Get output buffer of type T
    auto *outputBuffer = interpreter->typed_tensor<T>(outputTensorIndex);

    // copy output buffer to outputData
    unsigned long long totalElements = interpreter->tensor(outputTensorIndex)->bytes / sizeof(T);
    std::vector<T> outputData(outputBuffer, outputBuffer + totalElements);

    // Print inference ms in input inputData
    end = std::chrono::steady_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Inference Time in ms: " + std::to_string(inference_time) << std::endl;

    return outputData;
}


std::vector<YoloDetectionResult> Yolov5::nms(const std::vector<float> &outputdata) {

    // 获取输出矩阵维度信息
    TfLiteIntArray *output_dims = interpreter->tensor(interpreter->outputs()[0])->dims;

    const unsigned int num_anchors = output_dims->data[1]; // 输出的预测框个数
    const unsigned int num_classes = output_dims->data[2] - 5; // 类别个数

    const float *output_ptr = outputdata.data();
    // (cx, cy, w, h, conf, cls)
    std::vector<YoloDetectionResult> results;

    // 遍历锚点
    for (unsigned int i = 0; i < num_anchors; ++i) {
        // 获取当前锚点对应的输出行
        const float *row_ptr = output_ptr + i * (num_classes + 5);

        // 过滤低置信度的检测结果
        float obj_conf = row_ptr[4];
        if (obj_conf < obj_thres) continue; // filter first.

        // 获取类别置信度
        float max_cls_conf = 0.0;
        int best_cls_idx = 0;
        for (unsigned int c = 0; c < num_classes; ++c) {
            float cls_conf = row_ptr[5 + c];
            if (cls_conf > max_cls_conf) {
                max_cls_conf = cls_conf;
                best_cls_idx = c;
            }
        }

        if (max_cls_conf < cls_thres) continue; // class score.


        // 计算边界框坐标
        const float *offsets = row_ptr;
        float cx = offsets[0];
        float cy = offsets[1];
        float w = offsets[2];
        float h = offsets[3];

        // 填充人脸框信息
        YoloDetectionResult result;
        result.cx = cx;
        result.cy = cy;
        result.w = w;
        result.h = h;
        float x1 = (cx - w / 2.f);
        float y1 = (cy - h / 2.f);
        float x2 = (cx + w / 2.f);
        float y2 = (cy + h / 2.f);
        result.x1 = std::max(0.f, x1) * image_width;
        result.y1 = std::max(0.f, y1) * image_height;
        result.x2 = std::min(1.f, x2) * image_width;
        result.y2 = std::min(1.f, y2) * image_height;
        result.score = max_cls_conf * obj_conf;
        result.label = best_cls_idx;
        result.label_text = labels[best_cls_idx];
        result.flag = true;

        results.push_back(result);

    }


    // 执行非极大抑制
    std::sort(results.begin(), results.end(), [](const YoloDetectionResult &a, const YoloDetectionResult &b) {
        return a.score > b.score;
    });

    std::vector<YoloDetectionResult> nms_results;
    for (const YoloDetectionResult &result: results) {
        bool keep = true;
        for (const YoloDetectionResult &selected: nms_results) {
            float inter_x1 = std::max(result.x1, selected.x1);
            float inter_y1 = std::max(result.y1, selected.y1);
            float inter_x2 = std::min(result.x2, selected.x2);
            float inter_y2 = std::min(result.y2, selected.y2);

            float inter_width = std::max(0.0f, inter_x2 - inter_x1 + 1);
            float inter_height = std::max(0.0f, inter_y2 - inter_y1 + 1);

            float intersection = inter_width * inter_height;
            float area1 = (result.x2 - result.x1 + 1) * (result.y2 - result.y1 + 1);
            float area2 = (selected.x2 - selected.x1 + 1) * (selected.y2 - selected.y1 + 1);

            float iou = intersection / (area1 + area2 - intersection);

            if (iou > iou_thres) {
                keep = false;
                break;
            }
        }
        if (keep) {
            nms_results.push_back(result);
        }
    }

    return nms_results;
//    return results;
}


// 量化函数
template<typename T>
std::vector<T> Yolov5::quantize(const std::vector<float> &input, float scale, T zero_point) {
    std::vector<T> quantized(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        quantized[i] = static_cast<T>(round(input[i] / scale) + zero_point);
    }
    return quantized;
}

// 反量化函数
template<typename T>
std::vector<float> Yolov5::dequantize(const std::vector<T> &input, float scale, T zero_point) {
    std::vector<float> dequantized(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        dequantized[i] = (input[i] - zero_point) * scale;
    }
    return dequantized;
}

// 推理函数
std::vector<YoloDetectionResult> Yolov5::infer(const cv::Mat &image) {

    // Mat 转 vector
    std::vector<uchar> flatImage = matToVectorRGB(image);

    // 归一化
    std::vector<float> inputData = normalize(flatImage);

    // 前向传播
    int inputTensorIndex = interpreter->inputs()[0];
    std::vector<float> outputData;

    switch (input_type) {
        // 半精度量化和单精度
        case kTfLiteFloat32:
        case kTfLiteFloat16: {
            outputData = forward<float>(inputData);
            break;
        }

            // 无符号整形和整形量化
        case kTfLiteInt8:
        case kTfLiteUInt8: {
            TfLiteQuantization quantization_info = interpreter->tensor(inputTensorIndex)->quantization;
            if (quantization_info.type != kTfLiteAffineQuantization) {
                LOG(ERROR) << "Unsupported quantization type!";
                exit(-1);
            }
            TfLiteQuantizationParams *quant_params = static_cast<TfLiteQuantizationParams *>(quantization_info.params);
            float input_scale = quant_params->scale;
            int input_zero_point = quant_params->zero_point;

            if (input_type == kTfLiteInt8) {
                std::vector<int8_t> quantizedInput = quantize<int8_t>(inputData, input_scale, input_zero_point);
                std::vector<int8_t> quantizedOutput = forward<int8_t>(quantizedInput);
                outputData = dequantize<int8_t>(quantizedOutput, input_scale, input_zero_point);
            } else {
                std::vector<uint8_t> quantizedInput = quantize<uint8_t>(inputData, input_scale, input_zero_point);
                std::vector<uint8_t> quantizedOutput = forward<uint8_t>(quantizedInput);
                outputData = dequantize<uint8_t>(quantizedOutput, input_scale, input_zero_point);
            }
            break;
        }
            // 不支持的模型量化类型
        default:
            LOG(ERROR) << "Cannot handle input type "
                       << interpreter->tensor(inputTensorIndex)->type << " yet";
            exit(-1);
    }


    // 非极大抑制
    std::vector<YoloDetectionResult> results = nms(outputData);

    return results;
}


Yolov5::~Yolov5() {
    // 1. 移除 NNAPI 代理 (仅当不在 Windows 上)
#if USE_NNAPI
    TFLITE_MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(nullptr) == kTfLiteOk);
#endif
    // 2. 释放解释器和模型
    interpreter.reset();
    model.reset();

    // 3. 清除其他成员
    labels.clear();
}