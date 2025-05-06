using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace vsCallDll
{

    class TrtInfer:IDisposable
    {
        [DllImport("trt_one.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "initClsInfer")]
        private static extern IntPtr initClsInfer(string model_pth, StringBuilder msg);
        [DllImport("trt_one.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "initDetInfer")]
        private static extern IntPtr initDetInfer(string model_pth, StringBuilder msg);
        [DllImport("trt_one.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "initSegInfer")]
        private static extern IntPtr initSegInfer(string model_pth, StringBuilder msg);
        [DllImport("trt_one.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "destroyInfer")]
        private static extern void destroyInfer(IntPtr infer);

        [DllImport("trt_one.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "doInferenceByImgPath")]
        private static extern IntPtr doInferenceByImgPath(IntPtr infer, string img_pth, int[] roi, float conf_threshold, ref int det_num, StringBuilder msg);
        [DllImport("trt_one.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "doInferenceByCharArray")]
        private static extern IntPtr doInferenceByCharArray(IntPtr infer, IntPtr img_arr, int img_h, int img_w, float conf_threshold, ref int det_num, StringBuilder msg);
        [DllImport("trt_one.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "drawResult")]
        private static extern void drawResult(IntPtr infer, int stope_period, bool is_save);

        private IntPtr my_infer;
        private StringBuilder msg = new StringBuilder(10240);
        private int model_type;
        private int det_num;
        private IntPtr infer_res;

        public TrtInfer(string model_pth_, int model_type_=0)
        {
            model_type = model_type_;
            switch (model_type)
            {
                case 0:
                    my_infer = initClsInfer(model_pth_, msg);
                    break;
                case 1:
                    my_infer = initDetInfer(model_pth_, msg);
                    break;
                case 2:
                    my_infer = initSegInfer(model_pth_, msg);
                    break;
            }

        }

        public void doInferenceByImgPth(string img_pth)
        {
            infer_res = doInferenceByImgPath(my_infer, img_pth, null, 0.3f, ref det_num, msg);
            //drawResult(my_infer, 0, false);
        }

        public void doInferenceByUcharArr(string img_pth)
        {
            var img = Cv2.ImRead(img_pth);
            infer_res = doInferenceByCharArray(my_infer, img.Data, img.Rows, img.Cols, 0.3f, ref det_num, msg);
            //drawResult(my_infer, 0, false);

        }

        public void Dispose()
        {
            Console.WriteLine("Destruct...");
            destroyInfer(my_infer);
            GC.SuppressFinalize(this);
        }

        public void parseResult()
        {
            switch (model_type)
            {
                case 0:
                    {
                        CLS_RES[] results = new CLS_RES[det_num];
                        for(int i=0; i<det_num; i++)
                        {
                            IntPtr currentPtr = IntPtr.Add(infer_res, i * Marshal.SizeOf<CLS_RES>());
                            results[i] = Marshal.PtrToStructure<CLS_RES>(currentPtr);
                            Console.WriteLine($"Class_id: {results[i].cls} | Confidence: {results[i].confidence}");
                        }
                        break;
                    }
                case 1:
                    {
                        DET_RES[] results = new DET_RES[det_num];
                        for (int i = 0; i < det_num; i++)
                        {
                            IntPtr currentPtr = IntPtr.Add(infer_res, i * Marshal.SizeOf<DET_RES>());
                            results[i] = Marshal.PtrToStructure<DET_RES>(currentPtr);
                            Console.WriteLine($"Class_id: {results[i].cls} | Confidence: {results[i].confidence}");
                        }
                        break;
                    }
                case 2:
                    {
                        SEG_RES[] results = new SEG_RES[det_num];
                        for (int i = 0; i < det_num; i++)
                        {
                            IntPtr currentPtr = IntPtr.Add(infer_res, i * Marshal.SizeOf<SEG_RES>());
                            results[i] = Marshal.PtrToStructure<SEG_RES>(currentPtr);
                            Console.WriteLine($"Class_id: {results[i].cls} | Confidence: {results[i].confidence}");
                        }
                        break;
                    }
            }
        }

    }
}
