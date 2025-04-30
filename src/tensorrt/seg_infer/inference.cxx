#include "inference.h"
using namespace nvinfer1;
using std::vector;

int SEG_RES::get_area(){
    int width = std::abs(tl_x - br_x);
    int height = std::abs(tl_y - br_y);
    return width * height;
}

std::string SEG_RES::get_info(){
    std::stringstream ss;
    ss << "BBOX Coord: [" << tl_x << ", " << tl_y << ", " << br_x << ", " << br_y << "]";
    ss << " | BBOX Area: " << get_area();
    ss << " | Class_id: " << cls;
    ss << " | Confidence: " << confidence;
    return ss.str();
}

SEG_RES::~SEG_RES(){
    if(mask_data){
        delete[] mask_data;
        mask_data = nullptr;
    }
}


void printInfo(){
    std::cout << "TensorRT Segmentation Lib" << std::endl;
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
    std::cout << "[" << getTimeNow() << "] Use TensorRT to Segment objects\n";

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
        gEngine = static_cast<IRuntime*>(gRunTime)->deserializeCudaEngine(trtModelStream, size);
        gContext = static_cast<ICudaEngine*>(gEngine)->createExecutionContext();
        delete[] trtModelStream;
        warmUp(msg);
        std::cout << "init success" << std::endl;
    }catch(const std::exception& e){
        msg_ss << "[" << getTimeNow() << "] Error Message: " << e.what() << "\n";
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
    #ifdef DEBUG_TRT
        fs << msg_ss.str();
        fs.close();
    #endif
        return 1;
    }

    strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());
    #ifdef DEBUG_TRT
        fs << msg_ss.str();
        fs.close();
    #endif
    return 0;
}

SEG_RES* doInferenceByImgPth(const char* img_pth, const int* roi, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_TRT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceByImgPth> Func Image Path: " << img_pth << "\n";
    if(roi)
        msg_ss << "[" << getTimeNow() << "] ROI: [" << roi[0] << ", " << roi[1] << ", " << roi[2] << ", " << roi[3] << "]\n";

    try{
        cv::Mat img = cv::imread(img_pth, cv::IMREAD_COLOR);
        cv::Mat img_part;
    
        if(roi)
            img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
        else
            img.copyTo(img_part); 

    #ifdef DEBUG_TRT
        fs << msg_ss.str();
        fs.close();
    #endif
        return doInferenceByImgMat(img_part, score_threshold, det_num, msg);
    }catch(const std::exception& e){
        msg_ss << "[" << getTimeNow() << "] " << e.what() << "\n";
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
        #ifdef DEBUG_TRT
            fs << msg_ss.str();
            fs.close();
        #endif
        return new SEG_RES[1];
    }
}
    
SEG_RES* doInferenceBy3chImg(uchar* image_arr, const int height, const int width, const float score_threshold, int& det_num, char* msg){
#ifdef DEBUG_TRT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << "Got Image size: " << width << "x" << height << "\n";
#endif
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);

#ifdef DEBUG_TRT
    fs << "Convert Image size: " << img.size << "\n";
    fs.close();
#endif
    return doInferenceByImgMat(img, score_threshold, det_num, msg);
}
    
SEG_RES* doInferenceByImgMat(const cv::Mat& img_mat, const float score_threshold, int& det_num, char* msg){
#ifdef DEBUG_TRT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    std::stringstream msg_ss;
    msg_ss << "Call <doInferenceByImgMat> Func\n";
    
    try{ 
        if((gEngine == nullptr) | (gContext == nullptr)){
            throw std::runtime_error("No Valid Model");
        }
        
        void* buffers[3] = { NULL, NULL, NULL };  // 一个输入 两个输出
        std::vector<float> pred;
        std::vector<float> proto;
        cudaStream_t stream;

        int input_index = static_cast<ICudaEngine*>(gEngine)->getBindingIndex("images");  // 0
        int output0_index = static_cast<ICudaEngine*>(gEngine)->getBindingIndex("output0");  // 1
        int output1_index = static_cast<ICudaEngine*>(gEngine)->getBindingIndex("output1");  // 1
        msg_ss << "input_index: " << input_index << " output0: " << output0_index << " output1: " << output1_index << "\n";

        // 获取输入维度信息 NCHW
        int input_h = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(input_index).d[2];
        int input_w = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(input_index).d[3];
        msg_ss << "inputH: " << input_h << " inputW:" << input_w << "\n";

        // 获取输出维度信息 
        int pred_h = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(output0_index).d[1];
        int pred_w = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(output0_index).d[2];
        msg_ss << "pred data format: " << pred_h << "x" << pred_w << "\n";
        int proto_c = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(output1_index).d[1];
        int proto_h = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(output1_index).d[2];
        int proto_w = static_cast<ICudaEngine*>(gEngine)->getBindingDimensions(output1_index).d[3];
        msg_ss << "proto data format: " << proto_c << "x" << proto_h << "x" << proto_w << "\n";

        // 创建GPU显存输入 输出缓冲区
        cudaMalloc(&buffers[input_index], input_h * input_w * 3 * sizeof(float));
        cudaMalloc(&buffers[output0_index], pred_h * pred_w * sizeof(float));
        cudaMalloc(&buffers[output1_index], proto_c * proto_h * proto_w * sizeof(float));

        // 创建临时缓存输出
        pred.resize(pred_h * pred_w);
        proto.resize(proto_c * proto_h * proto_w);

        // 创建cuda流
        cudaStreamCreate(&stream);
        // 第一次推理12ms，后续的推理3ms左右
        cv::Mat tensor;
        preProcess(static_cast<ICudaEngine*>(gEngine), img_mat, tensor);

        // 内存到GPU显存
        cudaMemcpyAsync(buffers[input_index], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        // 推理
        static_cast<IExecutionContext*>(gContext)->enqueueV2(buffers, stream, nullptr);

        // GPU显存到内存
        cudaMemcpyAsync(pred.data(), buffers[output0_index], pred_h * pred_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(proto.data(), buffers[output1_index], proto_c * proto_h * proto_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

        // 后处理
        cv::Mat predmat(pred_h, pred_w, CV_32F, (float*)pred.data());
        predmat = predmat.t();
        cv::Mat protomat(proto_c, proto_h*proto_w, CV_32F, (float*)proto.data());
        // std::cout << "predmat: " << predmat.size << " protomat: " << protomat.size << std::endl;
        SEG_RES* res = postProcess(score_threshold, predmat, protomat, cv::Size(img_mat.cols, img_mat.rows), cv::Size(input_w, input_h), det_num);

        // 同步结束 释放资源
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        if (!buffers[0]) {
            delete[] buffers;
        }

        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());

        #ifdef DEBUG_TRT
            fs << msg_ss.str() << "\n";
            fs << "[" << getTimeNow() << "] ------------Inference Success.-------------------\n";
            fs.close();
        #endif    
        return res;
    }catch(const std::exception& e){
        msg_ss << "[" << getTimeNow() << "] ERROR Message: " << e.what() << "\n";
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
        #ifdef DEBUG_TRT
            fs << msg_ss.str();
            fs << "[" << getTimeNow() << "] ------------Inference Failed.-------------------\n";
            fs.close();
        #endif
        return new SEG_RES[1];
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
        cv::Mat blob_img = cv::Mat::ones(cv::Size(1024, 1024), CV_8UC3);
        int num;
        doInferenceByImgMat(blob_img, 0.5f, num, msg);
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

void preProcess(const ICudaEngine* engine, const cv::Mat& org_img, cv::Mat& blob){
    int input_h = engine->getBindingDimensions(0).d[2];
    int input_w = engine->getBindingDimensions(0).d[3];

    double org_h = org_img.rows, org_w = org_img.cols;
    // 前提假设，模型只有一个输入节点
    auto boarded_img = cv::Mat(cv::Size(input_w, input_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min((double)input_h/org_h, (double)input_w/org_w);
    cv::Mat resize_img;
    cv::resize(org_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));

    blob = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(input_w, input_h), cv::Scalar(0,0,0), true, false);
}

SEG_RES* postProcess(const float conf_threshold, const cv::Mat& pred_mat, const cv::Mat& proto_mat, const cv::Size& org_size, const cv::Size& infer_size, int& det_num){
    size_t pred_num = pred_mat.cols;

    vector<cv::Rect2d> boxes;
    vector<float> scores;
    vector<int> indices;
    vector<int> class_idx;
    vector<cv::Mat> masks;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.7f};

    for(int row=0; row<pred_mat.rows; ++row){
        const float* ptr = pred_mat.ptr<float>(row);
        vector<float> cls_conf = vector<float>(ptr+4, ptr+pred_num-32);
        cv::Point2i maxP;
        double maxV;
        cv::minMaxLoc(cls_conf, 0, &maxV, 0, &maxP);
        if (maxV<0.1) continue;  // 置信度非常低的直接跳过

        // 模型输出是 中心点 + 宽高
        cx = ptr[0], cy = ptr[1], w=ptr[2], h=ptr[3];  // Anchor Free 下 模型直接预测中心点和宽高
        boxes.emplace_back(cx-w/2, cy-h/2, w, h);

        cv::Mat m = pred_mat.row(row).colRange(pred_num-32, pred_num);
        masks.push_back(m);
        scores.push_back(static_cast<float>(maxV));
        class_idx.push_back(maxP.x);
    }
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);

    double r = std::min((double)infer_size.height/org_size.height, (double)infer_size.width/org_size.width);
    det_num = indices.size();
    SEG_RES* result = new SEG_RES[det_num];
    int counter{};
    for(auto it=indices.begin(); it!=indices.end(); ++it){
        float score = scores[*it];
        if (score < conf_threshold) continue;
        int cls = class_idx[*it];
        cv::Rect2d tmp = boxes[*it];
        tmp.x /= r;
        tmp.y /= r;
        tmp.width /= r;
        tmp.height /= r;

        cv::Mat m = masks[*it];
        cv::Mat mask = m * proto_mat;
        mask = mask.reshape(1, 160);

        int n_h = r*org_size.height, n_w = r*org_size.width;
        cv::resize(mask, mask, infer_size);
        mask = mask(cv::Rect(cv::Point(0,0), cv::Point(n_w, n_h)));
        cv::resize(mask, mask, org_size);

        cv::threshold(mask, mask, 0, 114, cv::THRESH_BINARY);  // 前景>0, 背景<0;
        cv::Mat instance_mask;
        mask(tmp).copyTo(instance_mask);

        result[counter].tl_x = tmp.tl().x;
        result[counter].tl_y = tmp.tl().y;
        result[counter].br_x = tmp.br().x;
        result[counter].br_y = tmp.br().y;
        result[counter].confidence = score;
        result[counter].cls = cls;
        result[counter].mask_h = instance_mask.rows;
        result[counter].mask_w = instance_mask.cols;
        result[counter].mask_type = instance_mask.type();
        result[counter].mask_data = new uchar[instance_mask.total()*instance_mask.elemSize()];
        memcpy(result[counter].mask_data, instance_mask.data, instance_mask.total()*instance_mask.elemSize());
        ++counter;
    }

    return result;
}
