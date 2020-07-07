#include "ObjDetect.h"
#include "log4cxx/Loging.h"
#include "BoostFun.h"
#include "MTCNN.h"
#include "algorithm"

using namespace cv;
using namespace std;
using namespace at;
using namespace chrono;
using namespace common_commonobj;
using namespace YOLOV3_ALG;

#define MODEL_PATH  ("./model/")

CObjDetect::CObjDetect():m_pCpuOrGpuDevice(NULL)
{
}

CObjDetect::~CObjDetect()
{
}

void CObjDetect::DetectOp(const cv::Mat& src, torch::Tensor& putout, std::vector<cv::Mat>& vecBuf, std::vector<float>& scale)
{
	//计时
	double t_start, t_end, t_cost;
	t_start = getTickCount();
	
	////缓存归一化后的帧
	Mat dst;
	cvtColor(src, dst, CV_BGR2RGB);											//bgr -> rgb
	resize(src, dst, Size(resize_width, resize_height));					//resize 图像
	dst.convertTo(dst, CV_32F, 1.0 / 255);								    //归一化到[0,1]区间
	vecBuf.emplace_back(dst);

	//获取帧的大小属性
	int frame_width = src.cols;
	int frame_height = src.rows;
	scale = { resize_width / frame_width, resize_height / frame_height };

	//图像预处理 注意需要和python训练时的预处理一致
	Mat resImage, traImage;
	//cvtColor(src, traImage, CV_BGR2RGB);
	//resize(traImage, resImage, Size(img_size, img_size));

	cv::Mat orig_img = src;
	cv::Mat resized_image;
	int img_w = frame_width;
	int img_h = frame_height;
	float w = (float)img_size;
	float h = (float)img_size;
	int new_w = img_w*min(w / img_w, h / img_h);
	int new_h = img_h*min(w / img_w, h / img_h);
	resize(src, resized_image, Size(new_w, new_h), cv::INTER_CUBIC);                  

	Mat canvas(img_size, img_size, CV_8UC3, cv::Scalar(128,128,128));

	auto t = (h - new_h) / 2;
	auto t1 = (h - new_h) / 2 + new_h;
	auto n = (w - new_w) / 2;
	auto n1 = (w - new_w) / 2 + new_w;

	for (int i = t; i < t1; i++)
	{
		for (int j = n; j < n1;j++)
		{
			canvas.at<cv::Vec3b>(i, j)[0] = resized_image.at<cv::Vec3b>(i, j)[0];
			canvas.at<cv::Vec3b>(i, j)[1] = resized_image.at<cv::Vec3b>(i, j)[1];
			canvas.at<cv::Vec3b>(i, j)[2] = resized_image.at<cv::Vec3b>(i, j)[2];
		}
	}
    
	cvtColor(canvas, traImage, CV_BGR2RGB);

	//resImage
	auto&& img_tensor = torch::from_blob(traImage.data, { 1, img_size, img_size, 3 }, at::kByte);//Mat转成tensor,大小为1,416,416,3  
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });						//调换顺序torch输入的格式1,3,416,416
	img_tensor = img_tensor.toType(at::kFloat).div_(255);
	img_tensor = img_tensor.to(*m_pCpuOrGpuDevice);							//预处理后的图像放入gpu

	//网络前向计算	
	auto&& output = module.forward({ img_tensor }).toTuple();				//前向传播获取结果
	
	//结果处理
	std::vector<torch::jit::IValue> results = output->elements();

	//低于置信度阈值(cls[:,:,4]为置信度值)的每个边界框的每个属性值（表示该边界框的一整行)都设为零
	auto cls = results[0].toTensor().cpu();
	cout << "cls: " << cls.size(0) <<" "<< cls.size(1) << " " << cls.size(2) << endl;
	auto conf_mask = cls.permute({ 2,0,1 })[s32ConfidenceIdx].gt(fConfidence).unsqueeze(2).to(at::kFloat);
	auto prediction = (cls * conf_mask);

	//用每个框的两个对角坐标计算两个框的IoU(框的(中心x,中心y,高度,宽度)属性转换成 (左上角 x, 左上角 y, 右下角 x, 右下角 y))
	auto prediction_viewer = prediction.accessor<float, 3>();
	auto pred_0 = prediction.size(0);
	auto pred_1 = prediction.size(1);
	auto pred_2 = prediction.size(2);

	for (int i = 0; i < pred_0; i++)
	{
		for (int j = 0; j < pred_1;j++)
		{
			prediction[i][j][0] = prediction_viewer[i][j][0] - prediction_viewer[i][j][2] / 2;
			prediction[i][j][1] = prediction_viewer[i][j][1] - prediction_viewer[i][j][3] / 2;
			prediction[i][j][2] = prediction_viewer[i][j][0] + prediction_viewer[i][j][2] / 2;
			prediction[i][j][3] = prediction_viewer[i][j][1] + prediction_viewer[i][j][3] / 2;
		}
	}

	//一次只能完成一张图像的置信度阈值设置和NMS(非极大值抑制)
	bool bFirstWrite = false;
	for (int i = 0; i < pred_0;i++)
	{
		//边框坐标、置信度、边框的分类最大分数及其对应索引的tensor拼接
		auto&& vecSplit = torch::split_with_sizes(prediction[0], { pred_2 - num_classes,num_classes }, 1);
		std::cout << "prediction[0]" << prediction[0].size(0) << " " << prediction[0].size(1) << std::endl;

		auto&& max = torch::max(vecSplit[1], 1);
		auto max_score = std::get<0>(max);
		max_score = max_score.unsqueeze(1);

		auto max_conf = std::get<1>(max).to(at::kFloat);
		max_conf = max_conf.unsqueeze(1);

		auto&& image_pred = torch::cat({vecSplit[0],max_score,max_conf }, 1);

		//置信度小于阈值的边界框行设为零,零的行丢弃
		auto&& non_zero_ind = torch::nonzero(image_pred.permute({ 1, 0 })[s32ConfidenceIdx]).squeeze();
		if (0 >= non_zero_ind.size(0))   //无检测结果的情况跳过对本图像的循环
		{
			continue;
		}
		auto&& image_pred_ = LCutTensorByIndexs(non_zero_ind, image_pred);


		//通过去重,获取一张图像中所检测到的类别(同一类别可能会有多个「真实」检测结果)
		vector<int> vecCls;
		auto&& img_classes = image_pred_.permute({ 1, 0 })[s32ClsIndx];
		for (int i = 0; i < img_classes.size(0); i++)
		{
			vecCls.push_back(img_classes[i].item().toInt());
		}
		sort(vecCls.begin(), vecCls.end());
		vecCls.erase(unique(vecCls.begin(), vecCls.end()),vecCls.end());

		//提取特定类别（用变量 cls 表示）的检测结果
		for (auto val : vecCls)
		{
			auto&& cls_mask = image_pred_ * (img_classes.eq(val).to(at::kFloat).unsqueeze(1));
			auto&& class_mask_ind = torch::nonzero(cls_mask.permute({ 1, 0 })[cls_mask.size(1) - 2]).squeeze();
			auto&& image_pred_class = LCutTensorByIndexs(class_mask_ind, image_pred_);

			auto conf_sort_index = std::get<1>(torch::sort(image_pred_class.permute({ 1, 0 })[s32ConfidenceIdx]));
			auto&& img_pred_cls = LCutTensorByIndexs(conf_sort_index, image_pred_class);
			img_pred_cls = img_pred_cls.to(at::kFloat);

			//NMS(非极大值抑制)
			auto idx = img_pred_cls.size(0);
			for (int i = 0; i < idx; i++)
			{
				//Get the IOUs of all boxes that come after the one we are looking at 
				int s32Len = idx - i - 1;
				if (0 >= s32Len)
				{
					break;
				}
				auto&& ious = GetBoxIou(img_pred_cls[i].unsqueeze(0), torch::split_with_sizes(img_pred_cls, { i + 1,s32Len })[1]);

				//Zero out all the detections that have IoU > treshhold
				auto&& iou_mask = ious.lt(fNmsThresh).to(at::kFloat).unsqueeze(1);
				for (int j = i + 1,k = 0;j < idx;j++,k++)
				{
					img_pred_cls[j] = (img_pred_cls[j] * iou_mask[k]);
				}
				
				//Remove the non-zero entries
				auto non_zero_ind = torch::nonzero(img_pred_cls.permute({ 1,0 })[s32ConfidenceIdx]).squeeze();
				image_pred_class = LCutTensorByIndexs(non_zero_ind, img_pred_cls);
			}

			//batch is identified by extra batch column
			auto batch_ind = torch::zeros({ image_pred_class.size(0), 1 }).fill_(i);
			auto out = torch::cat({ batch_ind,image_pred_class }, 1);
			(!bFirstWrite) ? putout = out : putout = torch::cat({ putout,out });
			bFirstWrite = true;
		}
	}


	//应该将方框的坐标转换为相对于填充后的图片中包含原始图片区域的计算方式
	auto&& im_dim = torch::tensor({ frame_width ,frame_height }).toType(at::kFloat).repeat({ 1,2 });
	im_dim = im_dim.repeat({ putout.size(0), 1 }).to(*m_pCpuOrGpuDevice);

	auto&& scaling_factor = std::get<0>(torch::min(img_size / im_dim, 1)).view({ -1, 1 }).to(*m_pCpuOrGpuDevice);

	/*
	 *将相对于输入网络图片(416x416)的方框属性变换成原图按照纵横比不变进行缩放后的区域的坐标
	 *scaling_factor*img_w和scaling_factor*img_h是图片按照纵横比不变进行缩放后的图片，即原图是768x576按照纵横比长边不变缩放到了416*372
	 *经坐标换算,得到的坐标还是在输入网络的图片(416x416)坐标系下的绝对坐标，但是此时已经是相对于416*372这个区域的坐标了，而不再相对于(0,0)原点
	 */
	auto&& permute = putout.permute({ 1,0 }).to(*m_pCpuOrGpuDevice);
	permute[1] -= ((img_size - scaling_factor * im_dim.permute({ 1,0 })[0].view({ -1, 1 })) / 2).squeeze();
	permute[3] -= ((img_size - scaling_factor * im_dim.permute({ 1,0 })[0].view({ -1, 1 })) / 2).squeeze();
	permute[2] -= ((img_size - scaling_factor * im_dim.permute({ 1,0 })[1].view({ -1, 1 })) / 2).squeeze();
	permute[4] -= ((img_size - scaling_factor * im_dim.permute({ 1,0 })[1].view({ -1, 1 })) / 2).squeeze();

	
	for (int i = 1; i < 6;i++)
	{
		permute[i] /= scaling_factor.squeeze();
	}

	putout = permute.permute({ 1,0 });

	for (int i = 0; i < putout.size(0);i++)  
	{
		putout[i][1] = torch::clamp(putout[i][1], 0.0, im_dim[i][0].item().toFloat());
		putout[i][3] = torch::clamp(putout[i][3], 0.0, im_dim[i][0].item().toFloat());

		putout[i][2] = torch::clamp(putout[i][2], 0.0, im_dim[i][1].item().toFloat());
		putout[i][4] = torch::clamp(putout[i][4], 0.0, im_dim[i][1].item().toFloat());
	}

	//运行时间计算
	t_end = getTickCount();
	t_cost = t_end - t_start;
	LOG_INFO("systerm") << string_format("time cost: %4.f ms\n", t_cost / 1000000.0);

	cout << "******putout*******" << putout << endl;
}

void CObjDetect::DetectFace()
{
#if 0
	testMultiThread();
#else
	//testSingleThread();
#endif
	
	//return;

	ENTER_FUNC;
	BTimer timer;

	std::string pnet_weight_path = std::string(MODEL_PATH) + "pnet.pt";
	std::string rnet_weight_path = std::string(MODEL_PATH) + "rnet.pt";
	std::string onet_weight_path = std::string(MODEL_PATH) + "onet.pt";

	TAlgParam alg_param;
	alg_param.min_face = 40;
	alg_param.scale_factor = 0.79;
	alg_param.cls_thre[0] = 0.6;
	alg_param.cls_thre[1] = 0.7;
	alg_param.cls_thre[2] = 0.7;

	TModelParam modelParam;
	modelParam.alg_param = alg_param;
	modelParam.model_path = { pnet_weight_path, rnet_weight_path, onet_weight_path };
	modelParam.mean_value = { { 127.5, 127.5, 127.5 },{ 127.5, 127.5, 127.5 },{ 127.5, 127.5, 127.5 } };
	modelParam.scale_factor = { 1.0f, 1.0f, 1.0f };
	modelParam.gpu_id = 0;
	modelParam.device_type = torch::DeviceType::CUDA;

	MTCNN mt;
	mt.InitDetector(&modelParam);
	std::string img_path = std::string(MODEL_PATH) + "../img/faces1.jpg";
	cv::Mat src = cv::imread(img_path);
	if (!src.data)
	{
		LOGE("cannot load image!");
		return;
	}

	std::vector<cv::Rect> outFaces;
	LOGI("warm up...");

	timer.reset();
	for (int i = 0; i < 5; i++)
	{
		mt.DetectFace(src, outFaces);
	}
	LOGI("warm up over, time cost: {}", timer.elapsed());

	timer.reset();
	mt.DetectFace(src, outFaces);
	LOGI(" cost: {}", timer.elapsed());

	for (auto& i : outFaces)
		cv::rectangle(src, i, { 0,255,0 }, 2);

	cv::imshow("result", src);
	cv::waitKey(100000);
	cv::imwrite("res2.jpg", src);
	LEAVE_FUNC;
}


int CObjDetect::testSingleThread()
{
	std::string pnet_weight_path = std::string(MODEL_PATH) + "pnet.pt";
	auto device_type = torch::DeviceType::CPU;

	ENTER_FUNC;
	BTimer timer;
	BTimer ti;

	auto p = torch::jit::load(pnet_weight_path);

	std::vector<torch::jit::IValue> inputs;
	auto input = torch::rand({ 1,3,1080,1920 });
	input = input.to(torch::Device(device_type, 0));
	inputs.emplace_back(input);
	LOGI("warm up...");

	timer.reset();
	for (int i = 0; i < 5; i++)
	{
		ti.reset();
		p.forward(inputs);
		LOGI("forward: {} ms", ti.elapsed());
	}
	LOGI("warm up over, time cost: {} ms", timer.elapsed());

	timer.reset();
	for (int i = 0; i < 50; i++)
	{
		ti.reset();
		p.forward(inputs);
		LOGI("forward: {} ms", ti.elapsed());
	}
	LOGI("run 50 iter, each iter mean time cost: {} ms", timer.elapsed() / 50.0);

	LEAVE_FUNC;
	return 0;
}

void CObjDetect::testMultiThread()
{
	ENTER_FUNC;
	int thread_num = 2;
	BTimer timer;
	BThreadPool t{ thread_num };

	std::vector<std::future<int>> res;
	timer.reset();
	for (int i = 0; i < thread_num; i++)
	{
		res.emplace_back(t.AddTask(&CObjDetect::testSingleThread,this));
	}
	for (auto& i : res)
	{
		i.get();
	}	
	LOGI("whole time cost: {} ms", timer.elapsed());

	LEAVE_FUNC;
}

//纵向截取指定多索引的tensor
torch::Tensor CObjDetect::LCutTensorByIndexs(const torch::Tensor & indexTensor, const torch::Tensor & inTensor, bool bView)
{
	auto indexTensor_size_0 = indexTensor.size(0);
	auto mid_tensor = inTensor[indexTensor[0].item().toInt()];
	for (int i = 1; i < indexTensor_size_0; i++)
	{
		mid_tensor = torch::cat({ mid_tensor,inTensor[indexTensor[i].item().toInt()] }, 0);
	}
	
	if (bView)
	{
		return mid_tensor.view({ -1,inTensor.size(1) });
	}
	
	return mid_tensor;
}

//获取box的iou
torch::Tensor CObjDetect::GetBoxIou(const torch::Tensor& box1, const torch::Tensor& box2)
{
	//Get the coordinates of bounding boxes
	auto b1_x1 = box1.permute({ 1,0 })[0];
	auto b1_y1 = box1.permute({ 1,0 })[1];
	auto b1_x2 = box1.permute({ 1,0 })[2];
	auto b1_y2 = box1.permute({ 1,0 })[3];

	auto b2_x1 = box2.permute({ 1,0 })[0];
	auto b2_y1 = box2.permute({ 1,0 })[1];
	auto b2_x2 = box2.permute({ 1,0 })[2];
	auto b2_y2 = box2.permute({ 1,0 })[3];

	//get the corrdinates of the intersection rectangle
	auto inter_rect_x1 = torch::max(b1_x1, b2_x1);
	auto inter_rect_y1 = torch::max(b1_y1, b2_y1);
	auto inter_rect_x2 = torch::max(b1_x2, b2_x2);
	auto inter_rect_y2 = torch::max(b1_y2, b2_y2);

	//Intersection area
	auto temp1 = inter_rect_x2 - inter_rect_x1 + 1;
	auto temp2 = torch::zeros({ inter_rect_x2.size(0) });
	auto temp31 = inter_rect_y2 - inter_rect_y1 + 1;
	auto temp32 = torch::zeros({ inter_rect_x2.size(0) });
	auto temp3 = torch::max(temp31, temp32);
	auto inter_area = torch::max(temp1, temp2*temp3);

	//Union Area
	auto b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1);
	auto b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1);
	return (inter_area / (b1_area + b2_area - inter_area));
}

void CObjDetect::DetectInit()
{
	if (!m_pCpuOrGpuDevice)
	{
		m_pCpuOrGpuDevice = new torch::Device(GetTorchDevice());
	}

	//init model	
	std::string strModelPath = std::string("./model/torch_script_eval.pt");
	module = torch::jit::load(strModelPath);
	module.to(*m_pCpuOrGpuDevice);
}