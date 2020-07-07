#pragma once
#include "TorchSource.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace SLOWFAST_ALG
{
	class CActionRecg
	{
	public:
		CActionRecg();
		~CActionRecg();


	    void RecgOp();
		void RecgFace();
		void RecgInit();
		void Test( torch::Tensor& putout, std::vector<cv::Mat>& vecBuf, std::vector<float>& scale);
		void TestVideo();
		void ReadImage(const cv::String IMAGE_PATH, torch::Tensor& image_batch);
		void ShowImg(cv::Mat& img, torch::Tensor& bbox, torch::Tensor& labels, torch::Tensor& probs, vector<int>& ids, int& count);
		template <typename T1, typename T2>
		void ToString(T1& input, T2& str);
		double round(double number, unsigned int bits);

	private:
		//测试单线程
		int testSingleThread();

		//测试多线程
		void testMultiThread();

		//纵向截取指定多索引的tensor
		TorchTensor LCutTensorByIndexs(const TorchTensor& indexTensor, const TorchTensor& inTensor, bool bview = true);

		//获取box_iou
		TorchTensor GetBoxIou(const TorchTensor& Box1, const TorchTensor& Box2);

	private:
		torch::Device* m_pCpuOrGpuDevice;

		int img_size = 224;
		int resize_width = 400;
		int resize_height = 300;
		int num_classes = 80;
		int s32ConfidenceIdx = 4; //?
		int s32ClsIndx = 6;//?
		float fConfidence = 0.5;
		float fNmsThresh = 0.4;

		torch::jit::script::Module module;
	};
}
