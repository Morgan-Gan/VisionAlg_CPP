#pragma once
#include "TorchSource.h"

namespace YOLOV3_ALG
{
	class CObjDetect
	{
	public:
		CObjDetect();
		~CObjDetect();

		void DetectOp(const cv::Mat& src,torch::Tensor& putout,std::vector<cv::Mat>& vecBuf,std::vector<float>& scale);
		void DetectFace();
		void DetectInit();

	private:
		//测试单线程
		int testSingleThread();

		//测试多线程
		void testMultiThread();

		//纵向截取指定多索引的tensor
		torch::Tensor LCutTensorByIndexs(const torch::Tensor& indexTensor, const torch::Tensor& inTensor,bool bView = true);

		//获取box_iou
		torch::Tensor GetBoxIou(const torch::Tensor& box1, const torch::Tensor& box2);

	private:
		torch::Device* m_pCpuOrGpuDevice;

		int img_size = 416;
		int resize_width = 400;
		int resize_height = 300;
		int num_classes = 80;
		int s32ConfidenceIdx = 4;
		int s32ClsIndx = 6;
		float fConfidence = 0.5;
		float fNmsThresh = 0.4;

		torch::jit::script::Module module;
	};
}
