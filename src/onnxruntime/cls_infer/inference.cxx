#include "inference.h"

void* gModelPtr = nullptr;

void printInfo(){
    std::cout << "OnnxRuntime Classification Inference Demo" << std::endl;
}

void run(const char* image_path, const char* onnx_path) {
    cv::Mat frame = cv::imread(image_path);
    int ih = frame.rows;
    int iw = frame.cols;

    // 创建InferSession, 查询支持硬件设备
    std::string onnxpath{onnx_path};
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    auto session_ptr = new Ort::Session(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Classification"), modelPath.c_str(), Ort::SessionOptions());

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    size_t numInputNodes = session_ptr->GetInputCount();
    size_t numOutputNodes = session_ptr->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 获取输入信息
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session_ptr->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session_ptr->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // 获取输出信息
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session_ptr->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[0]; // 1
    output_w = output_dims[1]; // 1000
    std::cout << "output format : HxW = " << output_h << "x" << output_w << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session_ptr->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

    // format frame
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session_ptr->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }

    // output data
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);

    cv::Point maxP;
    double maxScore;
    cv::minMaxLoc(dout, 0, &maxScore, 0, &maxP);
    std::cout << "cls: " << maxP.x << " score: " << maxScore << std::endl;

    // session_options.release();
    session_ptr->release();
    return ;
}

void initModel(const char* onnx_pth, char* msg){
    // OnnxRuntime 第一次推理和后续推理的耗时差不多，所以不做 WarmUp
    std::string onnxpath{onnx_pth};
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    try{
        gModelPtr = new Ort::Session(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Classification"), modelPath.c_str(), Ort::SessionOptions());
        std::cout << "Init Session Success." << std::endl;
        return;
    }catch(const std::exception& e){
        std::cout << e.what() << std::endl;
        return;
    }
}


CLS_RES doInferenceByImgPth(const char* image_pth, const int* roi, char* msg){
    std::stringstream msg_ss;
    CLS_RES result(-1, -1);

    try{
        cv::Mat img = cv::imread(image_pth);
        cv::Mat img_part;
        if(roi)
            img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
        else
            img.copyTo(img_part); 

        return doInferenceByImgMat(img_part, msg);
    }catch(const std::exception& e){
        msg_ss << e.what() << std::endl;
        std::cout << e.what() << std::endl;
    }
    return CLS_RES(-1, -1);
}

CLS_RES doInferenceBy3chImg(uchar* image_arr, const int height, const int width, char* msg){
    std::stringstream msg_ss;
    CLS_RES result(-1, -1);
    // 如果 width 和 height 与实际图像不符，出来的图像会扭曲，但不会报错
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return doInferenceByImgMat(img, msg);
}

CLS_RES doInferenceByImgMat(const cv::Mat& img_mat, char* msg){
#ifdef DEBUG_ORT_
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << "[" << getTimeNow() << "]" << "Call <doInferenceByImgMat> Func\n";
#endif

    std::stringstream msg_ss;
    CLS_RES result(-1, -1);

    try{

        Ort::Session* session_ptr = static_cast<Ort::Session*>(gModelPtr);

        if(session_ptr==nullptr)
            throw std::runtime_error("Error Model Pointer Convert Failed!");

        std::vector<std::string> input_node_names;
        std::vector<std::string> output_node_names;
        size_t numInputNodes = session_ptr->GetInputCount();

        size_t numOutputNodes = session_ptr->GetOutputCount();
        Ort::AllocatorWithDefaultOptions allocator;
        input_node_names.reserve(numInputNodes);

        // 获取输入信息
        int input_w{}, input_h{};
        for (int i = 0; i < numInputNodes; i++) {
            auto input_name = session_ptr->GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
            Ort::TypeInfo input_type_info = session_ptr->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            input_w = input_dims[3];
            input_h = input_dims[2];
            msg_ss << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
        }

        // 获取输出信息
        int output_h = 0;
        int output_w = 0;
        Ort::TypeInfo output_type_info = session_ptr->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_h = output_dims[0]; // 1
        output_w = output_dims[1]; // 1000
        msg_ss << "output format : HxW = " << output_h << "x" << output_w << std::endl;
        for (int i = 0; i < numOutputNodes; i++) {
            auto out_name = session_ptr->GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(out_name.get());
        }
        msg_ss << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

        // format frame
        cv::Mat blob = cv::dnn::blobFromImage(img_mat, 1.0 / 255.0, cv::Size(input_w, input_h), 0.0, true, false);
        size_t tpixels = input_h * input_w * 3;
        std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

        // set input data and inference
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
        const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
        const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
        std::vector<Ort::Value> ort_outputs;
        try {
            ort_outputs = session_ptr->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
        }
        catch (std::exception e) {
            msg_ss << e.what() << std::endl;
        }

        // output data
        const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
        cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);

        cv::Point maxP;
        double maxScore;
        cv::minMaxLoc(dout, 0, &maxScore, 0, &maxP);
        result.cls = maxP.x;
        result.confidence = maxScore;
        msg_ss << "cls: " << maxP.x << " score: " << maxScore << std::endl;
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
    }catch(const std::exception& e){
        msg_ss << e.what() << std::endl;
        strcpy_s(msg, msg_ss.str().length(), msg_ss.str().c_str());
    }

#ifdef DEBUG_ORT_
    fs << "[" << getTimeNow() << "]" << msg_ss.str() << "\n";
    if(fs.is_open())
        fs.close();
#endif

    return result;
}

void destroyModel(){
#ifdef DEBUG_ORT_
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    if(gModelPtr!=nullptr){
        Ort::Session* session_ptr = static_cast<Ort::Session*>(gModelPtr);
        session_ptr->release();        
    }
#ifdef DEBUG_ORT_
    fs << "[" << getTimeNow() << "]" << "Release Model Success.\n";
#endif
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
