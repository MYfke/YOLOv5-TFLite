#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Yolov5.h"

#define MODEL_FILE "/models/yolov5n-fp16.tflite"
#define LABEL_FILE "/models/yolo_label.txt"
#define IMAGE_DIR  "/images"

#include <windows.h>


cv::Mat resizeWithPadding(const cv::Mat &image, int targetWidth, int targetHeight) {
    cv::Mat resizedImage;

    // 计算缩放比例
    double scale = min(1.0 * targetWidth / image.cols, 1.0 * targetHeight / image.rows);

    // 缩放图片
    cv::resize(image, resizedImage, cv::Size(), scale, scale, cv::INTER_LINEAR);

    // 计算需要填充的空白区域大小
    int paddingX = targetWidth - resizedImage.cols;
    int paddingY = targetHeight - resizedImage.rows;

    // 创建填充后的图像
    cv::Mat paddedImage(targetHeight, targetWidth, resizedImage.type(), cv::Scalar(50, 50, 50));

    // 将缩放后的图像复制到填充后的图像中心
    cv::Rect roi((paddingX > 0 ? paddingX / 2 : 0), (paddingY > 0 ? paddingY / 2 : 0),
                 resizedImage.cols, resizedImage.rows);
    resizedImage.copyTo(paddedImage(roi));

    return paddedImage;
}

cv::Mat drawBoundingBoxes(cv::Mat image, const std::vector<YoloDetectionResult> &results) {
    for (const YoloDetectionResult &result: results) {
        cv::Point topLeft((int) (result.x1), (int) (result.y1));
        cv::Point bottomRight((int) (result.x2), (int) (result.y2)); // 右下角坐标

        cv::rectangle(image, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);

        // 添加标签文字
        std::string label = result.label_text;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 2;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
        cv::Point textOrg(topLeft.x,
                          topLeft.y - 10); // 10 pixels above the top left corner of the bounding box

        // 画一个背景矩形以增加文字的可读性
        cv::rectangle(image, textOrg + cv::Point(0, baseline),
                      textOrg + cv::Point(textSize.width, -textSize.height), cv::Scalar(255, 0, 0), -1);

        // 将标签文字放在矩形上
        cv::putText(image, label, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
    }
    return image;
}


int main(int argc, char **argv) {

    std::string modelFile = std::string(PROJECT_DIR) + std::string(MODEL_FILE);
    std::string labelFile = std::string(PROJECT_DIR) + std::string(LABEL_FILE);
    std::string imageDir = std::string(PROJECT_DIR) + std::string(IMAGE_DIR);

    // 创建模型
    Yolov5 model(modelFile, labelFile, TfLiteType::kTfLiteFloat16);

    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile((imageDir + "/*.jpg").c_str(), &findFileData);
    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to open directory." << std::endl;
        return 1;
    } else {
        do {
            std::string fileName = findFileData.cFileName;
            std::string imagePath = imageDir + "/" + fileName;

            cv::Mat ori_image = cv::imread(imagePath, cv::IMREAD_COLOR);

            if (!ori_image.empty()) {

                // 预处理图片
                cv::Mat image = resizeWithPadding(ori_image, 640, 640);
                // 执行推理
                std::vector<YoloDetectionResult> results = model.infer(image);
                // 绘制结果
                image = drawBoundingBoxes(image, results);

                cv::imshow("Image with Rectangles", image);
                cv::waitKey(0);
            }
        } while (FindNextFile(hFind, &findFileData) != 0);
        FindClose(hFind);
    }
    return 0;
}