#include "inference.h"



void printInfo(char* ret_msg, size_t msg_len){

    std::stringstream msg;
    auto backends = fastdeploy::GetAvailableBackends();
    msg << "Available Backends: \n";
    for(auto e:backends){
        msg << e << '\n';
    }    
    std::memcpy(ret_msg, msg.str().c_str(), msg_len);
}

// TODO: 待做batch推理

void initModel(const char* det_model_dir, const char* cls_model_dir, const char* rec_model_dir, const char* rec_dict_file, short backend_type, char* msg, size_t msg_len){
    fastdeploy::RuntimeOption option;
    option.UseCpu();
    if(backend_type==PADDLE){
        cout << "Use Paddle Backend\n";
        option.UsePaddleBackend();
    }else if(backend_type==ONNX){
        cout << "Use ONNX Backend\n";
        option.UseOrtBackend();
    }else if(backend_type==OPENVINO){
        cout << "Use OpenVINO Backend\n";
        option.UseOpenVINOBackend();
    }else{
        cout << "Error, Wrong Backend Type: " << backend_type << endl;
    }

    auto det_model_file = std::string(det_model_dir) + sep + "inference.pdmodel";
    auto det_params_file = std::string(det_model_dir) + sep + "inference.pdiparams";

    auto cls_model_file = std::string(cls_model_dir) + sep + "inference.pdmodel";
    auto cls_params_file = std::string(cls_model_dir) + sep + "inference.pdiparams";

    auto rec_model_file = std::string(rec_model_dir) + sep + "inference.pdmodel";
    auto rec_params_file = std::string(rec_model_dir) + sep + "inference.pdiparams";

    auto det_option = option;
    auto cls_option = option;
    auto rec_option = option;

    int cls_batch_size = 1;
    int rec_batch_size = 6;
    det_option.SetTrtInputShape("x", { 1, 3, 64, 64 }, { 1, 3, 640, 640 },
        { 1, 3, 960, 960 });
    cls_option.SetTrtInputShape("x", { 1, 3, 48, 10 }, { cls_batch_size, 3, 48, 320 },
        { cls_batch_size, 3, 48, 1024 });
    rec_option.SetTrtInputShape("x", { 1, 3, 48, 10 }, { rec_batch_size, 3, 48, 320 },
        { rec_batch_size, 3, 48, 2304 });

    det_model_ptr = new fastdeploy::vision::ocr::DBDetector(det_model_file, det_params_file, det_option);
    cls_model_ptr = new fastdeploy::vision::ocr::Classifier(cls_model_file, cls_params_file, cls_option);
    rec_model_ptr = new fastdeploy::vision::ocr::Recognizer(rec_model_file, rec_params_file, rec_dict_file, rec_option);

    assert(det_model_ptr->Initialized());
    assert(cls_model_ptr->Initialized());
    assert(rec_model_ptr->Initialized());

    // Parameters settings for pre and post processing of Det/Cls/Rec Models.
    // All parameters are set to default values.
    det_model_ptr->GetPreprocessor().SetMaxSideLen(960);
    det_model_ptr->GetPostprocessor().SetDetDBThresh(0.3);
    det_model_ptr->GetPostprocessor().SetDetDBBoxThresh(0.6);
    det_model_ptr->GetPostprocessor().SetDetDBUnclipRatio(1.5);
    det_model_ptr->GetPostprocessor().SetDetDBScoreMode("slow");
    det_model_ptr->GetPostprocessor().SetUseDilation(0);
    cls_model_ptr->GetPostprocessor().SetClsThresh(0.9);

    // The classification model is optional, so the PP-OCR can also be connected
    // in series as follows
    ppocr = new fastdeploy::pipeline::PPOCRv3{det_model_ptr, cls_model_ptr, rec_model_ptr};

    ppocr->SetClsBatchSize(cls_batch_size);
    ppocr->SetRecBatchSize(rec_batch_size);

    if (!ppocr->Initialized()) {
        std::cerr << "Failed to initialize PP-OCR." << std::endl;
        return;
    }

    cout << "Model Initialize Success." << endl;
}

void destroyModel(){
    if(det_model_ptr){
        delete det_model_ptr;
        det_model_ptr = nullptr;
    }

    if(cls_model_ptr){
        delete cls_model_ptr;
        cls_model_ptr = nullptr;
    }

    if(rec_model_ptr){
        delete rec_model_ptr;
        rec_model_ptr = nullptr;
    }

    if(ppocr){
        delete ppocr;
        ppocr = nullptr;
    }
}

OCR_RES doInferenceByImgPth(const char* image_pth, const double conf, char* msg, size_t msg_len){
    // cout << "Call doInferenceByImgPth Func !!" << endl;
    
    if(ppocr == nullptr){
        cout << "Error! model pointer is nullptr";
        return OCR_RES{};
    }
    cv::Mat im = cv::imread(image_pth);
    fastdeploy::vision::OCRResult res;
    OCR_RES result{};

    try{
        if(!ppocr->Predict(&im, &res)){
            std::cerr << "Failed to predict." << std::endl;
            return OCR_RES{};
        }

        // 解析ocr结果  bboxes rec words
        for(int i{}; i<res.rec_scores.size(); ++i){
            double rec_score = res.rec_scores[i];
            if(rec_score < conf)
                continue;
            
            OCR_ITEM item{};
            for(int c{}; c<res.boxes[i].size(); ++c)
                item.bbox[c] = res.boxes[i][c];

            memcpy_s(item.words, sizeof(item.words), res.text[i].c_str(), res.text[i].length());
            result.items.push_back(item);
        }    
        
        // // 结果可视化
        // auto vis_img = fastdeploy::vision::VisOcr(im, res, 0.8f);
        // cv::imshow("test", vis_img);
        // cv::waitKey(0);

    }catch(const std::exception& ex){
        std::cerr << ex.what() << endl;
    }
    
    return result;
}

OCR_RES doInferenceBy3chImg(uchar* image_arr, const int height, const int width, const double conf, char* msg, size_t msg_len){
    cout << "Call doInferenceBy3chImg Func !!" << endl;

    if(ppocr == nullptr){
        cout << "Error! model pointer is nullptr";
        return OCR_RES{};
    }
    cv::Mat im{cv::Size(width, height), CV_8UC3, image_arr};
    fastdeploy::vision::OCRResult res;
    OCR_RES result{};

    try{
        if(!ppocr->Predict(&im, &res)){
            std::cerr << "Failed to predict." << std::endl;
            return OCR_RES{};
        }

        // 解析ocr结果  bboxes rec words
        for(int i{}; i<res.rec_scores.size(); ++i){
            double rec_score = res.rec_scores[i];
            if(rec_score < conf)
                continue;
            
            OCR_ITEM item{};
            for(int c{}; c<res.boxes[i].size(); ++c)
                item.bbox[c] = res.boxes[i][c];

            memcpy_s(item.words, sizeof(item.words), res.text[i].c_str(), res.text[i].length());
            result.items.push_back(item);
        }    
        
        // // 结果可视化
        // auto vis_img = fastdeploy::vision::VisOcr(im, res, 0.8f);
        // cv::imshow("test", vis_img);
        // cv::waitKey(0);

    }catch(const std::exception& ex){
        std::cerr << ex.what() << endl;
    }
    
    return result;
}



