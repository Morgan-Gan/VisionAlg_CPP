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
	//打开视频文件
	VideoCapture capture("./video/test.mp4");

	//isOpen判断视频是否打开成功
	if (!capture.isOpened())
	{
		cout << "Movie open Error" << endl;
		return;
	}
	//获取视频帧频
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "帧率为:" << " " << rate << endl;
	cout << "总帧数为:" << " " << capture.get(CV_CAP_PROP_FRAME_COUNT) << endl;//输出帧总数
	Mat frame;
	namedWindow("Movie Player");

	double position = 0.0;
	//设置播放到哪一帧，这里设置为第0帧
	capture.set(CV_CAP_PROP_POS_FRAMES, position);
	while (1)
	{
		//读取视频帧
		if (!capture.read(frame))
			break;

		imshow("Movie Player", frame);
		//获取按键值
		char c = waitKey(33);
		if (c == 27)
			break;
	}
	capture.release();
	destroyWindow("Movie Player");
}

int main(int argc, char *argv[])
{
	DllParser dllParser;                                       //定义对象
	if (dllParser.Load(string("libVisionAlg.so")))             //加载库
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