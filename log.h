#ifndef TENSORFLOW_LITE_LOG_H_
#define TENSORFLOW_LITE_LOG_H_

#include <iostream>
#include <sstream>

#if defined(_WIN64)

#include <cstdio>

#elif defined(__ANDROID__)
#include <android/log.h>
#endif

namespace tflite_log {

    class Log {
        std::stringstream stream_;

    public:
        explicit Log(const char *severity) { stream_ << severity << ": "; }

        std::stringstream &Stream() { return stream_; }

        ~Log() {
#if defined(_WIN64)
            fprintf(stderr, "%s\n", stream_.str().c_str());
#elif defined(__ANDROID__)
            __android_log_print(ANDROID_LOG_DEBUG, "YOUR_TAG", "%s", stream_.str().c_str());
#endif
        }
    };

#define LOG(severity) tflite_log::Log(#severity).Stream()

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

}  // namespace tflite_log

#endif  // TENSORFLOW_LITE_LOG_H_
