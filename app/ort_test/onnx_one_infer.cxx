#include "base.h"
#include "interface.h"

void doInfer(Base* infer_ptr, const std::string img_pth, const short type=0){
    char msg[1024]{};
    int det_num{};
    void* result = infer_ptr->inferByImagePath(img_pth.c_str(), nullptr, 0.5f, det_num, msg);

    // cv::Mat img_mat = cv::imread(img_pth);
    // // cv::resize(img_mat, img_mat, cv::Size(), 0.5, 0.5);
    // void* result = infer_ptr->inferByCharArray(img_mat.data, img_mat.rows, img_mat.cols, 0.1f, det_num, msg);

    std::cout << "type: " << type << " det_num: " << det_num << std::endl;
    switch(type){
        case 0:{
            CLS_RES* cls_r = static_cast<CLS_RES*>(result);
            std::cout << cls_r[0].get_info() << std::endl;
            break; 
        }
        case 1:{
            DET_RES* det_r = static_cast<DET_RES*>(result);
            std::cout << "Detect: " << det_num << std::endl;
            for(int i{}; i<det_num; i++){
                std::cout << det_r[i].get_info() << std::endl;
            }
            break;
        }
        case 2:{
            SEG_RES* seg_r = static_cast<SEG_RES*>(result);
            std::cout << "Segment: " << det_num << std::endl;
            for(int i{}; i<det_num; i++){
                std::cout << seg_r[i].get_info() << std::endl;
            }
            break;
        }
    }

    if(det_num>0){
        infer_ptr->drawResult(true);
    }
}

void testBase(){
    char msg[1024]{};
    std::string cls_model_pth{R"(E:\Pretrained_models\YOLOv11-cls\yolo11n-cls.onnx)"};
    Classify cls_infer(cls_model_pth.c_str(), msg);

    std::string det_model_pth{R"(E:\Pretrained_models\YOLOv11\yolo11n.onnx)"};
    Detection det_infer(det_model_pth.c_str(), msg);

    std::string seg_model_pth{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-seg.onnx)"};
    Segmentation seg_infer(seg_model_pth.c_str(), msg);

    // std::string img_pth = R"(D:\share_dir\impression_detect\src\vs2015CallDll\test_images\for_seg\dogs.jpg)";
    std::string img_pth = R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)";

    for(int i{}; i<500; i++){
        std::cout << "---------------------- " << i << " ----------------------------" << std::endl;
        doInfer(&cls_infer, img_pth, 0);
        doInfer(&det_infer, img_pth, 1);
        doInfer(&seg_infer, img_pth, 2);
    }
}

void testInterface(){
    char msg[10240]{};

    std::string cls_model_pth{R"(E:\Pretrained_models\YOLOv11-cls\yolo11n-cls.onnx)"};
    auto cls_infer = static_cast<Classify*>(initClsInfer(cls_model_pth.c_str(), msg));

    // std::string cls_model_pth2{R"(D:\share_dir\impression_detect\src\vs2015CallDll\test_models\for_cls\best_fp16.engine)"};
    // auto cls_infer2 = static_cast<Classify*>(initClsInfer(cls_model_pth2.c_str(), msg));

    std::string det_model_pth{R"(E:\Pretrained_models\YOLOv11\yolo11n.onnx)"};
    auto det_infer = static_cast<Detection*>(initDetInfer(det_model_pth.c_str(), msg));

    std::string seg_model_pth{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-seg.onnx)"};
    auto seg_infer = static_cast<Segmentation*>(initSegInfer(seg_model_pth.c_str(), msg));

    int det_num{};
    std::string img_pth = R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)";
    std::string img_pth2 = R"(D:\share_dir\impression_detect\src\vs2015CallDll\test_images\for_cls\20250320163821925.jpg)";
    std::string img_pth3 = R"(E:\DataSets\bus.jpg)";

    auto cls_res = doInferenceByImgPath(cls_infer, img_pth3.c_str(), nullptr, 0.3f, det_num, msg);
    std::cout << static_cast<CLS_RES*>(cls_res)[0].get_info() << std::endl;

    // auto cls_res2 = doInferenceByImgPath(cls_infer2, img_pth3.c_str(), nullptr, 0.3f, det_num, msg);
    // std::cout << static_cast<CLS_RES*>(cls_res2)[0].get_info() << std::endl;

    auto det_res = doInferenceByImgPath(det_infer, img_pth3.c_str(), nullptr, 0.3f, det_num, msg);
    std::cout << static_cast<DET_RES*>(det_res)[0].get_info() << std::endl;

    auto seg_res = doInferenceByImgPath(seg_infer, img_pth3.c_str(), nullptr, 0.3f, det_num, msg);
    std::cout << static_cast<SEG_RES*>(seg_res)[0].get_info() << std::endl;


    drawResult(cls_infer, 1000);
    // drawResult(cls_infer2, 1000);
    drawResult(det_infer, 1000);
    drawResult(seg_infer, 0);

    destroyInfer(cls_infer);
    // destroyInfer(cls_infer2);
    destroyInfer(det_infer);
    destroyInfer(seg_infer);
}


int main(){

    testInterface();
    return 0;

    testBase();
    return 0;
} 