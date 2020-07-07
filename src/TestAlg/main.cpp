#include "main.h"
#include "DllParser.h"
#include "Any.h"
#include "log4cxx/Loging.h"
#include "TorchSource.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace common_template;
using namespace common_commonobj;

void PlayVideo()
{
	//����Ƶ�ļ�
	VideoCapture capture("./video/test.mp4");

	//isOpen�ж���Ƶ�Ƿ�򿪳ɹ�
	if (!capture.isOpened())
	{
		cout << "Movie open Error" << endl;
		return;
	}
	//��ȡ��Ƶ֡Ƶ
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "֡��Ϊ:" << " " << rate << endl;
	cout << "��֡��Ϊ:" << " " << capture.get(CV_CAP_PROP_FRAME_COUNT) << endl;//���֡����
	Mat frame;
	namedWindow("Movie Player");

	double position = 0.0;
	//���ò��ŵ���һ֡����������Ϊ��0֡
	capture.set(CV_CAP_PROP_POS_FRAMES, position);
	while (1)
	{
		//��ȡ��Ƶ֡
		if (!capture.read(frame))
			break;

		imshow("Movie Player", frame);
		//��ȡ����ֵ
		char c = waitKey(33);
		if (c == 27)
			break;
	}
	capture.release();
	destroyWindow("Movie Player");
}

int main(int argc, char *argv[])
{
	DllParser dllParser;                                       //�������
	if (dllParser.Load(string("libVisionAlg.so")))             //���ؿ�
	{
		using TupleType = std::tuple<>;
		TupleType&& Tuple = std::make_tuple();
		dllParser.ExcecuteFunc<bool(Any&&)>("InitModuleDll", std::move(Tuple));
		LOG_INFO("systerm") << "load libVisionAlg.so successful";
	}
	else
	{
		LOG_INFO("systerm") << "load libVisionAlg.so fail";
	}

	bool test = true;
	while (test)
	{
		using TupleProc = std::tuple<cv::Mat, torch::Tensor, torch::Tensor>;
		cv::Mat src;
		torch::Tensor Imgtensor;
		torch::Tensor Bbox;
		TupleProc&& tupleObj = std::make_tuple(src,Imgtensor, Bbox);

		dllParser.ExcecuteFunc<bool(Any&&)>("ProcModuleDll", std::move(tupleObj));
		LOG_INFO("systerm") << "ProcModuleDll function is done...";
		
		test = false;
	}
	return 0;
}