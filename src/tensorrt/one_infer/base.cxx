#include "base.h"

std::string CLS_RES::get_info(){
    std::stringstream ss;
    ss << "Class_id: " << cls << " | Confidence: " << confidence;
    return ss.str();
}

int DET_RES::get_area(){
    int width = std::abs(tl_x - br_x);
    int height = std::abs(tl_y - br_y);
    return width * height;
}

std::string DET_RES::get_info(){
    std::stringstream ss;
    ss << "BBOX Coord: [" << tl_x << ", " << tl_y << ", " << br_x << ", " << br_y << "]";
    ss << " | BBOX Area: " << get_area();
    ss << " | Class_id: " << cls;
    ss << " | Confidence: " << confidence;
    return ss.str();
}

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


class Logger: public ILogger{
    void log(Severity severity, const char* msg) noexcept{
        if(severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
}gLogger;

Base::Base(const char* model_pth_, char* msg):model_pth(model_pth_), my_msg(msg){
    // 加载模型文件
    std::ifstream file(model_pth, std::ios::binary);
    if (!file){
        // 模型文件读取失败
        throw std::runtime_error("Read Model Failed!");
    }
    file.seekg(0, std::ios::end);
    size_t model_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(model_size);
    file.read(engine_data.data(), model_size);
    file.close();
    
    runtime = createInferRuntime(gLogger);
    if(!runtime){
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    engine = runtime->deserializeCudaEngine(engine_data.data(), model_size);
    if(!engine){
        throw std::runtime_error("Failed to deserialize engine");
    }
    context = engine->createExecutionContext();
    if(!context){
        throw std::runtime_error("Failed to create execution context");
    }
}

Base::~Base(){
    if (!context) {
        context->destroy();
        context = nullptr;
    }
    if (!engine) {
        engine->destroy();
        engine = nullptr;
    }
    if (!runtime) {
        runtime->destroy();
        runtime = nullptr;
    }
    std::cout << "------------- Model Destroyed ------------------------\n";
}

Classify::Classify(const char* model_pth_, char* msg):Base(model_pth_, msg){
    // warmup

}

void* Classify::inferByMat(cv::Mat& img_mat, const float conf_threshold, int& num, char* msg) {

    checkImageChannel(img_mat);
    img_mat.copyTo(infer_img);

    if(result_ptr){
        delete[] static_cast<CLS_RES*>(result_ptr);
        result_ptr = nullptr;
        result_len = 0;
    }

    if((engine == nullptr) | (context == nullptr)){
        throw std::runtime_error("No Valid Model");
    }

    int input_index = engine->getBindingIndex("images");  // 0
    int output_index = engine->getBindingIndex("output0");  // 1        
    msg_ss << "input_index: " << input_index << " output_index: " << output_index << "\n";

    // 获取输入维度信息 NCHW
    int input_h = engine->getBindingDimensions(input_index).d[2];
    int input_w = engine->getBindingDimensions(input_index).d[3];
    msg_ss << "inputH: " << input_h << " inputW:" << input_w << "\n";

    // 获取输出维度信息 
    int output_h = engine->getBindingDimensions(output_index).d[0];
    int output_w = engine->getBindingDimensions(output_index).d[1];
    msg_ss << "output data format: " << output_h << "x" << output_w << "\n";

    // 创建GPU显存输入 输出缓冲区
    msg_ss << "input/output : " << engine->getNbBindings() << "\n"; // get the number of binding indices
    cudaMalloc(&io_buffer[input_index], input_h * input_w * 3 * sizeof(float));
    cudaMalloc(&io_buffer[output_index], output_h * output_w * sizeof(float));

    // 创建临时缓存输出
    outputHostBuffer = new float[output_h * output_w];

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 第一次推理12ms，后续的推理3ms左右
    cv::Mat tensor = cv::dnn::blobFromImage(img_mat, 1.0/255.0, cv::Size(input_w, input_h), 0.0, true, false);

    // 内存到GPU显存
    cudaMemcpyAsync(io_buffer[input_index], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 推理
    context->enqueueV2(io_buffer, stream, nullptr);

    // GPU显存到内存
    cudaMemcpyAsync(outputHostBuffer, io_buffer[output_index], output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 后处理
    cv::Mat probmat(output_h, output_w, CV_32F, outputHostBuffer);
    cv::Point maxL, minL;
    double maxv, minv;
    cv::minMaxLoc(probmat, &minv, &maxv, &minL, &maxL);
    int max_index = maxL.x;
    CLS_RES result(-1, -1);
    result.cls = max_index;
    result.confidence = maxv;

    // 同步结束 释放资源  TODO 测试放在实例属性， 推理完之后统一释放？？
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    for(int i{}; i<2; i++)
        cudaFree(io_buffer[i]);
    if(outputHostBuffer)
        delete[] outputHostBuffer;

    result_ptr = new CLS_RES[1]{result};
    result_len = 1;
    num = result_len;
    return result_ptr;
}

void* Classify::inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& num, char* msg){
    std::cout << ">>>>>>>>>>>>>>>>>>>>>> Classify infer by image path <<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
    cv::Mat img_part;
    if(roi)
        img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    else
        img.copyTo(img_part); 

    return inferByMat(img_part, conf_threshold, num, msg);
}

void* Classify::inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& num, char* msg){
    std::cout << "Classify infer by uchar array" << std::endl;
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return inferByMat(img, conf_threshold, num, msg);
}

Classify::~Classify(){
}

void Classify::drawResult(const short stop_period, const bool is_save)const {
    if(!result_ptr)
        return;

    int org_h = infer_img.rows;
    int org_w = infer_img.cols;

    cv::Mat new_img(org_h+30, org_w, CV_8UC3, cv::Scalar(0, 100, 200));
    infer_img.copyTo(new_img(cv::Rect(0, 0, org_w, org_h)));

    CLS_RES result = static_cast<CLS_RES*>(result_ptr)[0];
    std::string label{"Class_id: "};
    cv::putText(new_img, label+std::to_string(result.cls), cv::Point(0, new_img.rows-5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.0);
    cv::imshow("Show Result", new_img);
    cv::waitKey(stop_period);

    if(is_save){
        std::string save_file_name = getTimeNow() + ".jpg";
        std::replace(save_file_name.begin(), save_file_name.end(), ':', '-');
        std::replace(save_file_name.begin(), save_file_name.end(), ' ', '-');
        std::cout << "Save to: " << save_file_name << std::endl;
        if(cv::imwrite(save_file_name, new_img)){
            std::cout << "Save Image Success.\n";
        }else{
            std::cout << "Save Image Failed.\n";
        }
    }

}


Detection::Detection(const char* model_pth_, char* msg):Base(model_pth_, msg){
    // warmup

}

void* Detection::inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& det_num, char* msg){
    std::cout << ">>>>>>>>>>>>>>>>>>>>>> Detection infer by image path <<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    
    cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
    cv::Mat img_part;
    if(roi)
        img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    else
        img.copyTo(img_part); 

    return inferByMat(img_part, conf_threshold, det_num, msg);
}

void* Detection::inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& det_num, char* msg){
    std::cout << "Detection infer by uchar array" << std::endl;
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return inferByMat(img, conf_threshold, det_num, msg);
}

void* Detection::inferByMat(cv::Mat& img_mat, const float conf_threshold, int& det_num, char* msg) {
    
    checkImageChannel(img_mat);
    img_mat.copyTo(infer_img);

    if(result_ptr){
        delete[] static_cast<DET_RES*>(result_ptr);
        result_ptr = nullptr;
        result_len = 0;
    }

    if((engine == nullptr) | (context == nullptr)){
        throw std::runtime_error("No Valid Model");
    }

    int input_index = engine->getBindingIndex("images");  // 0
    int output_index = engine->getBindingIndex("output0");  // 1
    std::cout << "input_index: " << input_index << " output_index: " << output_index << "\n";

    // 获取输入维度信息 NCHW
    int input_h = engine->getBindingDimensions(input_index).d[2];
    int input_w = engine->getBindingDimensions(input_index).d[3];
    std::cout << "inputH: " << input_h << " inputW:" << input_w << "\n";

    // 获取输出维度信息 
    int output_h = engine->getBindingDimensions(output_index).d[1];
    int output_w = engine->getBindingDimensions(output_index).d[2];
    std::cout << "output data format: " << output_h << "x" << output_w << "\n";

    // 创建GPU显存输入 输出缓冲区
    cudaMalloc(&io_buffer[input_index], input_h * input_w * 3 * sizeof(float));
    cudaMalloc(&io_buffer[output_index], output_h * output_w * sizeof(float));

    // 创建零食缓存输出
    outputHostBuffer = new float[output_h * output_w];

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 第一次推理12ms，后续的推理3ms左右
    cv::Mat tensor;
    preProcess(img_mat, tensor);
    std::cout << "input mat: " << tensor.size << std::endl;

    // 内存到GPU显存
    cudaMemcpyAsync(io_buffer[input_index], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 推理
    context->enqueueV2(io_buffer, stream, nullptr);

    // GPU显存到内存
    cudaMemcpyAsync(outputHostBuffer, io_buffer[output_index], output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 后处理
    cv::Mat probmat(output_h, output_w, CV_32F, outputHostBuffer);
    double r = std::min((double)input_h/img_mat.rows, (double)input_w/img_mat.cols);
    result_ptr = postProcess(conf_threshold, probmat, r, det_num);
    result_len = det_num;

    // 同步结束 释放资源  TODO 测试放在实例属性， 推理完之后统一释放？？
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    for(int i{}; i<2; i++)
        cudaFree(io_buffer[i]);
    if(outputHostBuffer){
        delete[] outputHostBuffer;
    }
  
    return result_ptr;
}

void Detection::preProcess(const cv::Mat& org_img, cv::Mat& blob){
    int board_h = engine->getBindingDimensions(0).d[2];
    int board_w = engine->getBindingDimensions(0).d[3];

    double org_h = org_img.rows, org_w = org_img.cols;
    // 前提假设，模型只有一个输入节点
    auto boarded_img = cv::Mat(cv::Size(board_w, board_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min((double)board_h/org_h, (double)board_w/org_w);
    cv::Mat resize_img;
    cv::resize(org_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));
    blob = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(board_w, board_h), cv::Scalar(0,0,0), true, false);
}

DET_RES* Detection::postProcess(const float conf_threshold, cv::Mat& det_result_mat, const double& scale_ratio_, int& det_num){
    det_result_mat = det_result_mat.t();
    size_t pred_num = det_result_mat.cols;

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> class_idx;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.5f};

    for(int row=0; row<det_result_mat.rows; ++row){
        const float* ptr = det_result_mat.ptr<float>(row);
        std::vector<float> cls_conf = std::vector<float>(ptr+4, ptr+pred_num);
        cv::Point2i maxP;
        double maxV;
        cv::minMaxLoc(cls_conf, 0, &maxV, 0, &maxP);
        if (maxV<0.1) continue;  // 置信度非常低的直接跳过

        // 模型输出是 中心点 + 宽高
        cx = ptr[0], cy = ptr[1], w=ptr[2], h=ptr[3];  // 还不清楚是框架的问题还是有地方可以控制
        boxes.emplace_back(cx-w/2, cy-h/2, w, h);

        scores.push_back(static_cast<float>(maxV));
        class_idx.push_back(maxP.x);
    }
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);

    det_num = indices.size();
    DET_RES* result = new DET_RES[det_num];
    int counter{};
    for(auto it=indices.begin(); it!=indices.end(); ++it){
        float score = scores[*it];
        if (score < conf_threshold) continue;
        int cls = class_idx[*it];
        cv::Rect2d tmp = boxes[*it];
        tmp.x /= scale_ratio_;
        tmp.y /= scale_ratio_;
        tmp.width /= scale_ratio_;
        tmp.height /= scale_ratio_;

        result[counter].br_x = tmp.br().x;
        result[counter].br_y = tmp.br().y;
        result[counter].tl_x = tmp.tl().x;
        result[counter].tl_y = tmp.tl().y;
        result[counter].confidence = score;
        result[counter].cls = cls;
        ++counter;
    }

    return result;
}

Detection::~Detection(){

}

void Detection::drawResult(const short stop_period, const bool is_save) const {
    if(!result_ptr)
        return;

    cv::Mat new_img;
    infer_img.copyTo(new_img);
    auto box_color = cv::Scalar(0, 100, 200);
    auto font_color = cv::Scalar(200,200,200);
    for(int i{}; i<result_len; i++){
        DET_RES result = static_cast<DET_RES*>(result_ptr)[i];
        cv::Rect r(cv::Point(result.tl_x, result.tl_y), cv::Point(result.br_x, result.br_y));
        cv::rectangle(new_img, r, box_color, 1);
        std::stringstream label;
        label << "Class_id: " << result.cls << " conf: " << std::setprecision(2) << result.confidence;
        new_img(cv::Rect(cv::Point(result.tl_x, result.br_y), cv::Point(result.br_x, result.br_y+30))) = box_color;
        cv::putText(new_img, label.str(), cv::Point(result.tl_x, result.br_y+20), cv::FONT_HERSHEY_SIMPLEX, 0.6, font_color);
    }
    cv::imshow("Show Result", new_img);
    cv::waitKey(stop_period);

    if(is_save){
        std::string save_file_name = getTimeNow() + ".jpg";
        std::replace(save_file_name.begin(), save_file_name.end(), ':', '-');
        std::replace(save_file_name.begin(), save_file_name.end(), ' ', '-');
        std::cout << "Save to: " << save_file_name << std::endl;
        if(cv::imwrite(save_file_name, new_img)){
            std::cout << "Save Image Success.\n";
        }else{
            std::cout << "Save Image Failed.\n";
        }    
    }
}


Segmentation::Segmentation(const char* model_pth_, char* msg):Base(model_pth_, msg){
    // warmup

}

void* Segmentation::inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& det_num, char* msg){
    std::cout << ">>>>>>>>>>>>>>>>>>>>>> Segment infer by image path <<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    
    cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
    cv::Mat img_part;
    if(roi)
        img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    else
        img.copyTo(img_part); 

    
    return inferByMat(img_part, conf_threshold, det_num, msg);
}

void* Segmentation::inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& det_num, char* msg){
    std::cout << "Segmentation infer by uchar array" << std::endl;
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return inferByMat(img, conf_threshold, det_num, msg);
}

void* Segmentation::inferByMat(cv::Mat& img_mat, const float conf_threshold, int& det_num, char* msg) {
    
    checkImageChannel(img_mat);
    img_mat.copyTo(infer_img);
    
    if(result_ptr){
        delete[] static_cast<SEG_RES*>(result_ptr);
        result_ptr = nullptr;
        result_len = 0;
    }

    if((engine == nullptr) | (context == nullptr)){
        throw std::runtime_error("No Valid Model");
    }
    
    int input_index = engine->getBindingIndex("images");  // 0
    int output0_index = engine->getBindingIndex("output0");  // 1  检测结果
    int output1_index = engine->getBindingIndex("output1");  // 1  mask proto
    msg_ss << "input_index: " << input_index << " output0: " << output0_index << " output1: " << output1_index << "\n";

    // 获取输入维度信息 NCHW
    int input_h = engine->getBindingDimensions(input_index).d[2];
    int input_w = engine->getBindingDimensions(input_index).d[3];
    msg_ss << "inputH: " << input_h << " inputW:" << input_w << "\n";

    // 获取输出维度信息 
    int pred_h = engine->getBindingDimensions(output0_index).d[1];
    int pred_w = engine->getBindingDimensions(output0_index).d[2];
    msg_ss << "pred data format: " << pred_h << "x" << pred_w << "\n";
    int proto_c = engine->getBindingDimensions(output1_index).d[1];
    int proto_h = engine->getBindingDimensions(output1_index).d[2];
    int proto_w = engine->getBindingDimensions(output1_index).d[3];
    msg_ss << "proto data format: " << proto_c << "x" << proto_h << "x" << proto_w << "\n";

    // 创建GPU显存输入 输出缓冲区
    cudaMalloc(&io_buffer[input_index], input_h * input_w * 3 * sizeof(float));
    cudaMalloc(&io_buffer[output0_index], pred_h * pred_w * sizeof(float));
    cudaMalloc(&io_buffer[output1_index], proto_c * proto_h * proto_w * sizeof(float));

    // 创建临时缓存输出
    outputHostBuffer_1 = new float[pred_h * pred_w];
    outputHostBuffer_2 = new float[proto_c * proto_h * proto_w];

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 第一次推理12ms，后续的推理3ms左右
    cv::Mat tensor;
    preProcess(img_mat, tensor);

    // 内存到GPU显存
    cudaMemcpyAsync(io_buffer[input_index], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 推理
    context->enqueueV2(io_buffer, stream, nullptr);

    // GPU显存到内存
    cudaMemcpyAsync(outputHostBuffer_1, io_buffer[output0_index], pred_h * pred_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(outputHostBuffer_2, io_buffer[output1_index], proto_c * proto_h * proto_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 后处理
    cv::Mat predmat(pred_h, pred_w, CV_32F, outputHostBuffer_1);
    predmat = predmat.t();
    cv::Mat protomat(proto_c, proto_h*proto_w, CV_32F, outputHostBuffer_2);
    // std::cout << "predmat: " << predmat.size << " protomat: " << protomat.size << std::endl;
    result_ptr = postProcess(conf_threshold, predmat, protomat, cv::Size(img_mat.cols, img_mat.rows), cv::Size(input_w, input_h), det_num);
    result_len = det_num;

    // 同步结束 释放资源
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    for(int i{}; i<3; i++)
        cudaFree(io_buffer[i]);
    if(outputHostBuffer_1){
        delete[] outputHostBuffer_1;
    }
    if(outputHostBuffer_2){
        delete[] outputHostBuffer_2;
    }
    return result_ptr;
}

void Segmentation::preProcess(const cv::Mat& org_img, cv::Mat& blob){
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

SEG_RES* Segmentation::postProcess(const float conf_threshold, const cv::Mat& pred_mat, const cv::Mat& proto_mat, const cv::Size& org_size, const cv::Size& infer_size, int& det_num){
    size_t pred_num = pred_mat.cols;

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> class_idx;
    std::vector<cv::Mat> masks;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.5f};

    for(int row=0; row<pred_mat.rows; ++row){
        const float* ptr = pred_mat.ptr<float>(row);
        std::vector<float> cls_conf = std::vector<float>(ptr+4, ptr+pred_num-32);
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

    // TODO: 有BUG
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

        cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);  // 前景>0, 背景<0;
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

Segmentation::~Segmentation(){

}

void Segmentation::drawResult(const short stop_period, const bool is_save)const{
    if(!result_ptr)
        return;

    cv::Mat new_img;
    infer_img.copyTo(new_img);
    auto box_color = cv::Scalar(0, 100, 200);
    auto font_color = cv::Scalar(200,200,200);
    auto mask_color = cv::Scalar(0, 10, 255);
    SEG_RES* seg_res_ptr = static_cast<SEG_RES*>(result_ptr);
    for(int i{}; i<result_len; i++){
        cv::Rect r(cv::Point(seg_res_ptr[i].tl_x, seg_res_ptr[i].tl_y), cv::Point(seg_res_ptr[i].br_x, seg_res_ptr[i].br_y));
        cv::Mat patch(new_img(r));
        cv::Mat binary_mask(seg_res_ptr[i].mask_h, seg_res_ptr[i].mask_w, seg_res_ptr[i].mask_type, seg_res_ptr[i].mask_data);
        binary_mask.convertTo(binary_mask, CV_8U);
        cv::Mat color_mask;
        cv::cvtColor(binary_mask, color_mask, cv::COLOR_GRAY2BGR);
        color_mask.setTo(mask_color, binary_mask==255);

        cv::resize(binary_mask, binary_mask, cv::Size(patch.cols, patch.rows));
        cv::resize(color_mask, color_mask, cv::Size(patch.cols, patch.rows));
        cv::Mat tmp;
        cv::addWeighted(patch, 0.5, color_mask, 0.5, 0, tmp);
        tmp.copyTo(patch, binary_mask);

        cv::rectangle(new_img, r, box_color, 1);
        std::stringstream label;
        label << "Class_id: " << seg_res_ptr[i].cls << " conf: " << std::setprecision(2) << seg_res_ptr[i].confidence;
        new_img(cv::Rect(cv::Point(seg_res_ptr[i].tl_x, seg_res_ptr[i].br_y), cv::Point(seg_res_ptr[i].br_x, seg_res_ptr[i].br_y+30))) = box_color;
        cv::putText(new_img, label.str(), cv::Point(seg_res_ptr[i].tl_x, seg_res_ptr[i].br_y+20), cv::FONT_HERSHEY_SIMPLEX, 0.6, font_color);
    }
    cv::imshow("Show Result", new_img);
    cv::waitKey(stop_period);

    if(is_save){
        std::string save_file_name = getTimeNow() + ".jpg";
        std::replace(save_file_name.begin(), save_file_name.end(), ':', '-');
        std::replace(save_file_name.begin(), save_file_name.end(), ' ', '-');
        std::cout << "Save to: " << save_file_name << std::endl;
        if(cv::imwrite(save_file_name, new_img)){
            std::cout << "Save Image Success.\n";
        }else{
            std::cout << "Save Image Failed.\n";
        }    
    }
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

void checkImageChannel(cv::Mat& input_mat){
    if(input_mat.channels()>3){
        throw std::runtime_error("Error, image has more than 3 channels");
    }else if(input_mat.channels()==1){
        cv::cvtColor(input_mat, input_mat, cv::COLOR_GRAY2BGR);
    }
}