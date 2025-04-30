#include "inference.h"
using namespace nvinfer1;

void printInfo(){
    std::cout << "TensorRT CLassification Lib" << std::endl;
}

class Logger: public ILogger{
    void log(Severity severity, const char* msg) noexcept{
        if(severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
}gLogger;

void run(){

    std::string enginepath = R"(E:\le_trt\models\resnet18.engine)";
    std::ifstream file(enginepath, std::ios::binary);
    char* trtModelStream = nullptr;
    int size = 0;
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    auto runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    auto engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    auto context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    void* buffers[2] = { NULL, NULL };  // ????
    std::vector<float> prob;
    cudaStream_t stream;

    int input_index = engine->getBindingIndex("input");  // 0
    int output_index = engine->getBindingIndex("output");  // 1
    std::cout << "input_index: " << input_index << " output_index: " << output_index << std::endl;

    // 获取输入维度信息 NCHW
    int input_h = engine->getBindingDimensions(input_index).d[2];
    int input_w = engine->getBindingDimensions(input_index).d[3];
    std::cout << "inputH: " << input_h << " inputW:" << input_w << std::endl;

    // 获取输出维度信息 
    int output_h = engine->getBindingDimensions(output_index).d[0];
    int output_w = engine->getBindingDimensions(output_index).d[1];
    std::cout << "output data format: " << output_h << "x" << output_w << std::endl;

    // 创建GPU显存输入 输出缓冲区
    std::cout << "input/output : " << engine->getNbBindings() << std::endl; // get the number of binding indices
    cudaMalloc(&buffers[input_index], input_h * input_w * 3 * sizeof(float));
    cudaMalloc(&buffers[output_index], output_h * output_w * sizeof(float));

    // 创建零食缓存输出
    prob.resize(output_h * output_w);

    // 创建cuda流
    cudaStreamCreate(&stream);

    // 第一次推理12ms，后续的推理3ms左右
    for(int i{}; i<10; ++i){
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat image = cv::imread(R"(E:\le_trt\models\dog.jpg)");
        cv::Mat rgb, blob;
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, blob, cv::Size(224, 224));
        blob.convertTo(blob, CV_32F);
        blob = blob / 255.0;
        cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
        cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);

        // HWC -> CHW
        cv::Mat tensor = cv::dnn::blobFromImage(blob);

        // 内存到GPU显存
        cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        // 推理
        context->enqueueV2(buffers, stream, nullptr);

        // GPU显存到内存
        cudaMemcpyAsync(prob.data(), buffers[1], output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

        // 后处理
        cv::Mat probmat(output_h, output_w, CV_32F, (float*)prob.data());
        cv::Point maxL, minL;
        double maxv, minv;
        cv::minMaxLoc(probmat, &minv, &maxv, &minL, &maxL);
        int max_index = maxL.x;
        std::cout << "label id: " << max_index << " score: " << maxv << std::endl;

        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> spend = stop - start;
        std::cout << " Inference cost: " << spend.count() << "ms" << std::endl;
    }

    // 同步结束 释放资源
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (!context) {
        context->destroy();
    }
    if (!engine) {
        engine->destroy();
    }
    if (!runtime) {
        runtime->destroy();
    }
    if (!buffers[0]) {
        delete[] buffers;
    }

    std::cout << "Job Done" << std::endl;

    return;
}

void* gRunTime = nullptr;
void* gEngine = nullptr;
void* gContext = nullptr;

int initModel(const char* model_pth, char* msg){    
    std::stringstream msg_ss;
#ifdef DEBUG_TRT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    std::cout << "[" << getTimeNow() << "] Use TensorRT to Classify image\n";

    msg_ss << "[" << getTimeNow() << "] Call <initModel> Func Model Path: " << model_pth << "\n";

    try{
        const char* sufix = std::strrchr(model_pth, '.');
        if(std::strcmp(sufix, ".engine")!=0){
            throw std::runtime_error("Only Support .engine Model File.");
        }

        // 加载模型文件
        std::ifstream file(model_pth, std::ios::binary);
        char* trtModelStream = nullptr;
        int size = 0;
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            file.read(trtModelStream, size);
            file.close();
        }

        if (size == 0){
            // 反序列化模型失败
            throw std::runtime_error("Read Model Failed!");
        }
        
        gRunTime = createInferRuntime(gLogger);
        // TODO: 如果读取模型识别后，捕获不到异常
        gEngine = static_cast<IRuntime*>(gRunTime)->deserializeCudaEngine(trtModelStream, size);

        gContext = static_cast<ICudaEngine*>(gEngine)->createExecutionContext();
        delete[] trtModelStream;
        msg_ss << "[" << getTimeNow() << "] Init Model Success" << std::endl;
    }catch(const std::exception& e){
        msg_ss << "[" << getTimeNow() << "] Error Message: " << e.what() << "\n";
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
    #ifdef DEBUG_TRT
        fs << msg_ss.str();
        fs.close();
    #endif
        return 1;
    }

    warmUp(msg);
    strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
    #ifdef DEBUG_TRT
        fs << msg_ss.str() << "\n";
        fs.close();
    #endif
    return 0;
}

CLS_RES doInferenceByImgPth(const char* image_pth, const int* roi, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_TRT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceByImgPth> Func Image Path: " << image_pth << "\n";
    if(roi)
        msg_ss << "[" << getTimeNow() << "] ROI: [" << roi[0] << ", " << roi[1] << ", " << roi[2] << ", " << roi[3] << "]\n";
    
    try{
        cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
        cv::Mat img_part;
        if(roi)
            img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
        else
            img.copyTo(img_part); 

        #ifdef DEBUG_TRT
            fs << msg_ss.str();
            fs.close();
        #endif
        return doInferenceByImgMat(img_part, msg);

    }catch(const std::exception& e){
        msg_ss << "[" << getTimeNow() << "] " << e.what() << "\n";
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
        #ifdef DEBUG_TRT
            fs << msg_ss.str();
            fs.close();
        #endif
        return CLS_RES(-1, -1);
    }
}
    
CLS_RES doInferenceBy3chImg(uchar* image_arr, const std::int32_t height, const std::int32_t width, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_TRT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceBy3chImg> Func Image Size: " << height << "x" << width << "\n";
    
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);

#ifdef DEBUG_TRT
    fs << msg_ss.str();
    fs.close();
#endif
    return doInferenceByImgMat(img, msg);
}

CLS_RES doInferenceByImgMat(const cv::Mat& img_mat, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_TRT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceByImgMat> Func\n";
    
    CLS_RES ret(-1, -1);
    try{
        if((gEngine == nullptr) | (gContext == nullptr)){
            throw std::runtime_error("No Valid Model");
        }

        void* buffers[2] = { NULL, NULL };  // 一个输入CUDA  一个输出CUDA
        std::vector<float> prob;
        cudaStream_t stream;

        int input_index = static_cast<ICudaEngine*>(gEngine)->getBindingIndex("images");  // 0
        int output_index = static_cast<ICudaEngine*>(gEngine)->getBindingIndex("output0");  // 1
        msg_ss << "input_index: " << input_index << " output_index: " << output_index << "\n";

        // 获取输入维度信息 NCHW
        int input_h = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(input_index).d[2];
        int input_w = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(input_index).d[3];
        msg_ss << "inputH: " << input_h << " inputW:" << input_w << "\n";

        // 获取输出维度信息 
        int output_h = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(output_index).d[0];
        int output_w = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(output_index).d[1];
        msg_ss << "output data format: " << output_h << "x" << output_w << "\n";

        // 创建GPU显存输入 输出缓冲区
        msg_ss << "input/output : " << static_cast<ICudaEngine*>(gEngine)->getNbBindings() << "\n"; // get the number of binding indices
        cudaMalloc(&buffers[input_index], input_h * input_w * 3 * sizeof(float));
        cudaMalloc(&buffers[output_index], output_h * output_w * sizeof(float));

        // 创建零食缓存输出
        prob.resize(output_h * output_w);

        // 创建cuda流
        cudaStreamCreate(&stream);

        // 第一次推理12ms，后续的推理3ms左右
        cv::Mat tensor = cv::dnn::blobFromImage(img_mat, 1.0/255.0, cv::Size(input_w, input_h), 0.0, true, false);

        // 内存到GPU显存
        cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        // 推理
        static_cast<IExecutionContext*>(gContext)->enqueueV2(buffers, stream, nullptr);

        // GPU显存到内存
        cudaMemcpyAsync(prob.data(), buffers[1], output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

        // 后处理
        cv::Mat probmat(output_h, output_w, CV_32F, (float*)prob.data());
        cv::Point maxL, minL;
        double maxv, minv;
        cv::minMaxLoc(probmat, &minv, &maxv, &minL, &maxL);
        int max_index = maxL.x;
        ret.cls = max_index;
        ret.confidence = maxv;

        // 同步结束 释放资源
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        if (!buffers[0]) {
            delete[] buffers;
        }

        #ifdef DEBUG_TRT
            fs << msg_ss.str() << "\n";
            fs << "[" << getTimeNow() << "] ------------Inference Success.-------------------\n";
            fs.close();
        #endif    
        return ret;
    }catch(const std::exception& e){
        msg_ss << "[" << getTimeNow() << "] ERROR: " << e.what();
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
        #ifdef DEBUG_TRT
            fs << msg_ss.str();
            fs << "[" << getTimeNow() << "] ------------Inference Failed.-------------------\n";
            fs.close();
        #endif
        return ret;
    }
}


int destroyModel(){
    // TODO  感觉没有用。。。
    if (!gContext) {
        static_cast<IExecutionContext*>(gContext)->destroy();
        gContext = nullptr;
    }
    if (!gEngine) {
        static_cast<ICudaEngine*>(gEngine)->destroy();
        gEngine = nullptr;
    }
    if (!gRunTime) {
        static_cast<IRuntime*>(gRunTime)->destroy();
        gRunTime = nullptr;
    }

    std::cout << "Model Destroyed." << std::endl;
    return 0;
}


void warmUp(char* msg){
    std::stringstream msg_ss;
    msg_ss << "Call <warmUp> Func ...\n";

    try{        
        cv::Mat blob_img = cv::Mat::ones(cv::Size(1024, 1024), CV_32FC3);
        char msg[1024];
        doInferenceByImgMat(blob_img, msg);
        msg_ss << "Inference Done\n" << "WarmUp Complete";
    }catch(std::exception ex){
        msg_ss << "Catch Error in Warmup Func\n";
        msg_ss << "Error Message: " << ex.what() << std::endl;
    }

    strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());
}


std::string getTimeNow() {
    std::stringstream ss;
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    
    // 将时间点转换为time_t以便进一步转换为本地时间
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    
    // 转换为本地时间并打印
    ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %X");
    return ss.str(); 
}
