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
		//���Ե��߳�
		int testSingleThread();

		//���Զ��߳�
		void testMultiThread();

		//�����ȡָ����������tensor
		torch::Tensor LCutTensorByIndexs(const torch::Tensor& indexTensor, const torch::Tensor& inTensor,bool bView = true);

		//��ȡbox_iou
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
