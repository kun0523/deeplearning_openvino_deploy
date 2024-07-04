

// 1. 前处理  resize + normalize + CHW + 
// 1.1 resize 
// 2. 模型输出
// 3. 后处理


// ocr det preprocess
// 1. resize: 960 / max(h,w)
// 2. normalize:  (img/255 - mean) / std
// 3. HWC -> CHW
// 4. 保存 （图片原始尺寸， 缩放比例）

// ocr rec preprocess
// 0. 图像尺寸 3*48*320
// 1. 前面检测到多个，可以组成batch一次推理
// 2. 同一各个bbox的长宽； 保持原bbox wh_ratio  基础 h=48  resize w
// 3. 归一化： HWC -> CHW  (img_mat/255 - 0.5)/0.5
// 4. padding 到 3*48*320

// ocr rec postprocess
// 0. 输出 batch_num * 40 * 97(英文字库)
// 1. 对 97 个类取最大的类， batch_num * 40;
// 2. 去重



// TODO: 实现个bbox推理
// TODO: 实现多个bbox batch 推理
