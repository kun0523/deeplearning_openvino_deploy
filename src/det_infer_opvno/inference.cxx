#include "inference.h"


void* initModel(const char* model_path){

    try{
        ov::Core core;
        // core.set_property("CPU", ov::hint::inference_precision(ov::element::i8));
        std::string xml_pth{model_path};
        std::string bin_pth{model_path};
        xml_pth += "/best.xml";
        bin_pth += "/best.bin";
        std::shared_ptr<ov::Model> model = core.read_model(xml_pth, bin_pth);

        std::cout << "input size: " << model->inputs().size() << std::endl;
        std::cout << "output size: " << model->outputs().size() << std::endl;

        
        // -------- Step 3. Set up input

        // Read input image to a tensor and set it to an infer request
        // without resize and layout conversions
        cv::Mat img = cv::imread(R"(E:\DataSets\dents_det\org_D1\gold_scf\cutPatches640\NG\4_5398.jpg)");

        ov::element::Type input_type = ov::element::u8;
        ov::Shape input_shape = {1, static_cast<unsigned>(img.rows), static_cast<unsigned>(img.cols), 3};
        // std::shared_ptr<unsigned char> input_data = ;

        // just wrap image data by ov::Tensor without allocating of new memory
        ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, img.data);

        const ov::Layout tensor_layout{"NHWC"};

        // -------- Step 4. Configure preprocessing --------

        ov::preprocess::PrePostProcessor ppp(model);

        // 1) Set input tensor information:
        // - input() provides information about a single model input
        // - reuse precision and shape from already available `input_tensor`
        // - layout of data is 'NHWC'
        ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
        // 2) Adding explicit preprocessing steps:
        // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
        // - apply linear resize from tensor spatial dims to model spatial dims
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        // 4) Suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout("NCHW");
        // 5) Set output tensor information:
        // - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(ov::element::f32);

        // 6) Apply preprocessing modifying the original 'model'
        model = ppp.build();

        // -------- Step 5. Loading a model to the device --------
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");        
        auto inference_precision = compiled_model.get_property(ov::hint::inference_precision);

        // -------- Step 6. Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // -------- Step 7. Prepare input --------
        infer_request.set_input_tensor(input_tensor);

        // -------- Step 8. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 9. Process output
        const ov::Tensor& output_tensor = infer_request.get_output_tensor();

        ov::Shape shape = output_tensor.get_shape();  // 1*5*8400
        size_t rank = shape.size();
        size_t batch_size = shape[0]; // 几个batch
        size_t col_num = shape[1];  // 预测了几个值
        size_t row_num = shape[2];  // 预测了几个框

        const float* output_buff = output_tensor.data<const float>();
        cv::Mat m = cv::Mat(cv::Size(col_num, row_num), CV_32F, const_cast<float*>(output_buff));

        vector<DET_RES> out_res_vec;
        // YOLOV10 需要指定 false  YOLOV8 v11 指定 true；
        postProcess(0.5, m, 1.0, 0, 0, 11, out_res_vec);
    

        for(int i{}; i< row_num; ++i){
            auto row_ptr = output_tensor.data<float>() + i*col_num;

            for(int j{}; j<col_num; ++j){
                std::cout << row_ptr[j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;


    }catch(const std::exception& ex){
        std::cout << ex.what() << std::endl;
    }

    return nullptr;
}

char* postProcess(const float conf_threshold, cv::Mat& det_result_mat, const double scale_ratio_, const int left_padding, const int top_padding, short model_type, std::vector<DET_RES>& out_vec){

    if(model_type!=8 && model_type!=10 && model_type != 11){
        throw std::runtime_error("Error Model Type " + std::to_string(model_type) + " Only support Model Type: v8 v10 v11.");
    }

    // if(model_type!=10)
    //     det_result_mat = det_result_mat.t();    
    size_t pred_num = det_result_mat.cols;

    vector<cv::Rect2d> boxes;
    vector<float> scores;
    vector<int> indices;
    vector<int> class_idx;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.5f};

    for(int row=0; row<det_result_mat.rows; ++row){
        const float* ptr = det_result_mat.ptr<float>(row);
        vector<float> cls_conf = vector<float>(ptr+4, ptr+pred_num);
        cv::Point2i maxP;
        double maxV;
        cv::minMaxLoc(cls_conf, 0, &maxV, 0, &maxP);
        if (maxV<0.1) continue;  // 置信度非常低的直接跳过

        switch (model_type)
        {
        case 8:
        case 10:
            // 模型输出是 两个角点坐标
            tl_x = ptr[0], tl_y = ptr[1], br_x=ptr[2], br_y=ptr[3];
            boxes.emplace_back(tl_x, tl_y, br_x-tl_x, br_y-tl_y);
            break;        

        case 11:
            // 模型输出是 中心点 + 宽高
            cx = ptr[0], cy = ptr[1], w=ptr[2], h=ptr[3];  // 还不清楚是框架的问题还是有地方可以控制
            boxes.emplace_back(cx-w/2, cy-h/2, w, h);
            break;
        }
        scores.push_back(static_cast<float>(maxV));
        class_idx.push_back(maxP.x);
    }

    switch(model_type){
        case 8:
        case 11:
            // 使用NMS过滤重复框
            cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);
            break;

        case 10:
            // 不需要使用NMS过滤
            indices = vector<int>(boxes.size());
            std::iota(indices.begin(), indices.end(), 0);
            break;
    }           

    for(auto it=indices.begin(); it!=indices.end(); ++it){
        float score = scores[*it];
        if (score < conf_threshold) continue;
        int cls = class_idx[*it];
        cv::Rect2d tmp = boxes[*it];
        tmp.x -= left_padding;
        tmp.x /= scale_ratio_;
        tmp.y -= top_padding;
        tmp.y /= scale_ratio_;
        tmp.width /= scale_ratio_;
        tmp.height /= scale_ratio_;
        out_vec.emplace_back(tmp, cls, score);     
    }

    return "Post Process Complete.";
}


DET_RES* doInferenceByImgPth(const char* img_pth, void* model_ptr, const int* roi, const float score_threshold, const short model_type, size_t& det_num, char* msg){
    return nullptr;
}