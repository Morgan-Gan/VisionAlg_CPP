#include "ActionRecg.h"
#include "log4cxx/Loging.h"
#include "BoostFun.h"
#include "TorchSource.h"
#include <deque>
#include <algorithm>
#include "iomanip"    //保留小数

using namespace cv;
using namespace std;
using namespace at;
using namespace chrono;
using namespace common_commonobj;
using namespace SLOWFAST_ALG;

#define MODEL_PATH ("./model")

CActionRecg::CActionRecg():m_pCpuOrGpuDevice(NULL)
{
}

CActionRecg::~CActionRecg()
{

}

//const Mat& src, TorchTensor& putout, const MatVec& vecBuf, const FloatVec& scale
void CActionRecg::RecgOp()
{
	//计时
	double t_start, t_end, t_cost;
	t_start = getTickCount();

	//获得image_batch
	cv::String Image_path = "./img/brick1/*.jpg";
	torch::Tensor Image_batch;
	ReadImage(Image_path, Image_batch);
	cout << Image_batch.size(0) << " " << Image_batch.size(1) << " " << Image_batch.size(2) << " " << Image_batch.size(3) << " " << Image_batch.size(4) << endl;



	t_end = getTickCount();
	t_cost = t_end - t_start;
	LOG_INFO("slowfast_alg") << string_format("Loading data cost %4.f ms\n", t_cost/1000000.0);
	
}


void CActionRecg::ReadImage(const cv::String IMAGE_PATH, torch::Tensor& image_batch)
{

	vector<cv::String> fn;
	glob(IMAGE_PATH, fn, false);
	deque<Mat> vecBuf;
	size_t count = fn.size();

	int count_n = 0;
	for (size_t i = 0; i < count; i++)
	{
		Mat src, dst;
		src = cv::imread(fn[i]);
		if(!src.data)
		{
			LOG_ERROR("slowfast_alg") << "Cannot load image";
			return;
		}

		//bgr ->rgb
		cvtColor(src, dst, CV_BGR2RGB);
		//resize图像
		resize(src, dst, Size(resize_width, resize_height));
		//归一化到[0,1]区间
		dst.convertTo(dst, CV_32F, 1.0 / 255);

		//获取帧的大小属性
		int frame_width = src.cols;
		int frame_height = src.rows;
		auto scale = { resize_width / frame_width, resize_height / frame_height };

		//图像预处理，注意需要与python训练时的预处理一致，传给网络
		Mat resImage, tarImage;
		cvtColor(src, tarImage, CV_BGR2RGB);
		resize(tarImage, resImage, Size(img_size, img_size));

		if (vecBuf.size() < 64)
		{
			vecBuf.emplace_back(dst);
		}
		else
		{
			vecBuf.pop_front();
			vecBuf.push_back(dst);
		}
		
		if (vecBuf.size()==64)
		{
			if (count_n%3==0)
			{
				//vecBuf由deque转成tensor，并完成拼接：64 224 224 3
				int s32Size = vecBuf.size();
				auto img_tensor = torch::from_blob(vecBuf.at(0).data, { 1, img_size, img_size, 3 }, at::kByte);
				for (int i = 1; i < s32Size;i++)
				{
					auto&& tensor_ = torch::from_blob(vecBuf.at(i).data, { 1, img_size, img_size, 3 }, at::kByte);
					img_tensor = torch::cat({ img_tensor ,tensor_ }, 0);
				}

				//调换顺序torch输入格式3,64,224,224并升一维为1,3,64,224,224
				img_tensor = img_tensor.permute({ 3,0,1,2 });
				image_batch = img_tensor.unsqueeze(0);
				image_batch = image_batch.toType(at::kFloat).div_(255);
				image_batch = image_batch.to(*m_pCpuOrGpuDevice);

				//前向传播获取结果
				auto detector_bboxes = torch::rand({ 1,3,4 });
				detector_bboxes = detector_bboxes.to(*m_pCpuOrGpuDevice);
				auto&& output = module.forward({ image_batch,detector_bboxes }).toTuple();
				//结果处理
				std::vector<torch::jit::IValue> result = output->elements();

				auto detection_bboxs = result[0].toTensor().cpu();
				auto detection_cls = result[1].toTensor().cpu();
				auto detection_probs = result[2].toTensor().cpu();
				
			}
			count_n += 1;
		}		
	}
}

void CActionRecg::ShowImg(cv::Mat& img, torch::Tensor& bbox, torch::Tensor& labels, torch::Tensor& probs, vector<int>& ids, int& count)
{

	   int count_2 = 0;
	   int labels_r = labels.size(0);
	   for (size_t j = 0; j < labels_r; j++)
	   {

		  auto real_x_min = bbox.permute({ 1,0 })[0][j].item().toDouble()+1;       //.t()
		  auto real_y_min = bbox.permute({ 1,0 })[1][j].item().toDouble()+1;
		  auto real_x_max = bbox.permute({ 1,0 })[2][j].item().toDouble()+10;
		  auto real_y_max = bbox.permute({ 1,0 })[3][j].item().toDouble()+10;

		  auto id = ids[j];
		  string id_str;
		  ToString(id, id_str);                            // 函数模板:2 自动类型 推导(1 显示类型 调用)

		  //在每一帧上画矩形：frame帧，四个坐标参数，颜色，宽度
		  cv::rectangle(img, Rect(real_x_min * 10, real_y_min * 10, real_x_max * 50, real_y_max * 50), { 225, 0, 0 }, 4);

		  //读标签,打开文件流
		  ifstream fin("./label/ava_action_list_v2.0.csv");
		  string line, label_id, label_name, label_type;
		  vector<string> fields;                            //声明一个字符串向量
		  int n = 0;

		  while (getline(fin, line))						//整行读取，遇到文件尾efo终止
		  {
			  istringstream sin(line);						//将整行字符串line读到字符串流istringstream中
			  string field;
			  while (getline(sin, field, ','))				//将字符串流sin中的字符读入到field字符
			  {
				  fields.push_back(field);
			  }

		  }

		  int label_c = labels.size(1);
		  for (size_t k = 0; k < label_c;k++)
		  {
			 count_2 = count_2 + 1;
			 auto label = labels[j][k];
			 auto prob = probs[j][k].item().toDouble();
			 prob = round(prob, 3);

			  //取标签，画文本框
			 string label_str, prob_str;
			 ToString(label, label_str);
			 for (vector<string>::iterator it = fields.begin(); it != fields.end(); it++)
			  {
				  if (*it == label_str)
				  {
					  label_id = *it;
					  label_name = *(it + 1);
				  }
			  }

			 ToString(prob, prob_str);
			  string text = label_name + ":" + prob_str;         //setprecision(3)		取三位有效数字

			  Point origin;
			  origin.x = real_x_min*2*(j+1) * 10 + 15;
			  origin.y = real_y_max*2*(j+1) * 10 - 15 * count_2;
			  Point origin1;
			  origin1.x = real_x_min * 2 * (j + 1) * 10 + 10;
			  origin1.y = real_y_max * 2 * (j + 1) * 10 + 20;

			  cv::putText(img, text, origin, FONT_HERSHEY_COMPLEX, 0.5, { 0,0,255 }, 1);
			  cv::putText(img, "id:"+id_str, origin1, FONT_HERSHEY_COMPLEX, 0.5, { 0,0,255 }, 1);

		  }
   }
	 //显示的时候打开
	//cv::imshow("img", img);
	//cv::waitKey(1000000);

	//保存图像，int转换string
	string DetImg_PATH = "./det/";
	string output = DetImg_PATH + "det" + ".jpg";   
	cv::imwrite(output, img);
}


void CActionRecg::Test(torch::Tensor& putout, std::vector<cv::Mat>& vecBuf, std::vector<float>& scale)
{
	char *imageSrc = "./img/brick1/100.jpg";
	Mat&& src = imread(imageSrc, -1);
	IplImage *iplImage = cvLoadImage(imageSrc, -1);
	if (src.data == 0 || iplImage->imageData == 0)
	{
		LOG_DEBUG("slowfast_alg") << "Load image fail\n";
		return;
	}

	//process data
	std::vector<torch::jit::IValue> input11;
	std::vector<torch::jit::IValue> input22;

	auto input1 = torch::rand({ 1,3,64,320,460 });
	auto input2 = torch::rand({ 1,3,4 });

	input1 = input1.to(*m_pCpuOrGpuDevice);
	input2 = input2.to(*m_pCpuOrGpuDevice);

	auto&& output = module.forward({ input1, input2 }).toTuple();   
	std::vector<torch::jit::IValue> result = output->elements();

	auto detection_bboxs = result[0].toTensor().cpu();
	auto detection_cls = result[1].toTensor().cpu();
	auto detection_probs = result[2].toTensor().cpu();

	LOG_INFO("slowfast_alg") << string_format("detection_bboxs: %d %d |detection_cls: %d %d |detection_probs size: %d %d", detection_bboxs.size(0), detection_bboxs.size(1), detection_cls.size(0), detection_cls.size(1), detection_probs.size(0), detection_probs.size(1));

	//for test
	scale = { 0.2, 0.2 };  
	auto deb_0 = detection_bboxs.size(0);
	auto deb_1 = detection_bboxs.size(1);
	for (int i = 0; i < deb_0; i++)
	{
		detection_bboxs[i][0] /= scale[0];
		detection_bboxs[i][2] /= scale[0];
		detection_bboxs[i][1] /= scale[1];
		detection_bboxs[i][3] /= scale[1];
	}

	int count = 17;
	vector<int> ids = { 1,2,3};

	CActionRecg show;
	show.ShowImg(src,detection_bboxs, detection_cls, detection_probs, ids, count);
}

template <typename T1, typename T2>
void CActionRecg::ToString(T1& input, T2& str)               // T2=string
{
	stringstream ss;
	ss << input;
	ss >> str;
}
void CActionRecg::TestVideo()
{
	//打开视频文件
	VideoCapture capture;
	Mat frame;
	std::cout << "Movie1 load success" << std::endl;
	frame=capture.open("./video/helmet6.avi");
	
	//isOpen判断视频是否打开成功
	if (!capture.isOpened())
	{
		std::cout << "Movie open Eror" << std::endl;
		return;  //状态码,-1为错，0为对
	}

	//获取视频帧数
	double rate = capture.get(CV_CAP_PROP_FPS);
	std::cout << "帧率为：" << " " << rate << std::endl;
	std::cout << "总帧数为：" << " " << capture.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;
	namedWindow("Movie Player");

	/*设置播放到哪一帧，这里设置第0帧*/
	double position = 0.0;
	capture.set(CV_CAP_PROP_POS_FRAMES, position);
	while (1)
	{
		//读取视频帧
		if (!capture.read(frame))
			break;

		imshow("Move Player", frame);

		//获取按键值ESC(ASCII码为27)
		char c = waitKey(33);
		if (c == 27)
		{
			break;
		}
			
	}

	capture.release();
	destroyWindow("Move Player");
}

double CActionRecg::round(double number, unsigned int bits)
{
	stringstream ss;
	ss << fixed << setprecision(bits) << number;
	ss >> number;
	return number;
}

void CActionRecg::RecgInit()
{
	if (!m_pCpuOrGpuDevice)
	{
		m_pCpuOrGpuDevice = new torch::Device(GetTorchDevice());
	}

	//init model
	std::string strModelPath = std::string("./model/slowfast_50_eval_three.pt");
	module = torch::jit::load(strModelPath);
	module.to(*m_pCpuOrGpuDevice);
	
	LOG_INFO("slowfast_alg") << string_format("Loading model successful %s", strModelPath);

}