#include "AlgModuleMng.h"
#include "MsgBusMng.h"
#include "CfgMng.h"
#include "BoostFun.h"
#include "TorchSource.h"
#include "log4cxx/log4cxx.h"

using namespace at;
using namespace cv;
using namespace std;
using namespace chrono;
using namespace common_commonobj;
using namespace common_template;
using namespace VISION_ALG;

const string strAlgModuleNode("AlgModule");

bool CAlgModuleMng::LoadAlgModule()
{
	//获取配置对象
	nlohmann::json&& jsCfg = SCCfgMng.GetJsonCfg();
	for (auto it = jsCfg[strAlgModuleNode].begin(); it != jsCfg[strAlgModuleNode].end(); ++it)
	{
		string strModuleType(it.key());
		for (auto elm : it.value())
		{
			string&& strElmName = elm["name"];
			string&& strModuleName = string("./lib") + strElmName + string(".so");

			if (!InitAlgModule(strModuleType,strModuleName, std::move(elm)))
			{
				return false;
			}
		}
	}

	return true;
}

bool CAlgModuleMng::OperateAlgModule()
{
	double t_start, t_cost, t_end;
	t_start = getTickCount();

	//cv::String&& src_path = "./img/brick1/*.jpg";
	cv::String&& src_path = "./img/person/*.jpg";
	vector<cv::String> fn;
	glob(src_path, fn, false);
	deque<Mat> vecBuf;
	size_t count = fn.size();

	int count_n = 0;
	for (size_t i = 0; i < count;i++)
	{
		//Mat src, dst;
		Mat&& src = cv::imread(fn[i]);
		if (!src.data)
		{
			LOG_ERROR("systerm") << "Load image fail!";
			return false;
		}

		//调用yolov3消息处理函数
		TorchTensor output;
		MatVec VecBuf;
		FloatVec vecScale;
	
		string&& strTopic("Yolov3ProcMatExt");
		SCMsgBusMng.GetMsgBus()->SendReq<bool, const Mat&, TorchTensor&, MatVec&, FloatVec&>(src, output, VecBuf, vecScale, strTopic);

	}
	
	//TorchTensor output1;
	//MatVec vecBuf1;
	//FloatVec vecScale1;

	////向某个主题发送消息，需要主题和消息类型。消息总线接收到消息后会找到并通知对的消息处理函数
	//string&& strTopic("SlowFastProcMatExt");
	//SCMsgBusMng.GetMsgBus()->SendReq<bool, TorchTensor&, MatVec&, FloatVec&>(output1, vecBuf1, vecScale1, strTopic);
	//LOG_INFO("systerm") << string_format("vecScale size: %d\n", vecScale1.size());


	//for (auto dll : m_mapDllShrPtr)
	//{
	//	if ("ObjDetectModule./libYolov3.so" == dll.first)
	//	{
	//		Mat&& src = cv::imread("./img/dog.jpg");
	//		TorchTensor output;
	//		MatVec vecBuf;
	//		FloatVec vecScale;

	//		string&& strTopic("Yolov3ProcMatExt");
	//		SCMsgBusMng.GetMsgBus()->SendReq<bool, const Mat&, TorchTensor&, MatVec&, FloatVec&>(src, output, vecBuf, vecScale, strTopic);

	//		LOG_INFO("systerm") << string_format("output size : %d %d |vecBuf size : %d |vecScale size : %d\n", output.size(0), output.size(1), vecBuf.size(), vecScale.size());
	//	}

	//}

	t_end = getTickCount();
	t_cost = t_start - t_end;
	LOG_INFO("systerm") << string_format("Process data cost %4.f ms\n", t_cost);
	return true;
}


bool CAlgModuleMng::LoadData(cv::Mat src, torch::Tensor imgtensor)
{

	
	cout << "This is LoadData...." << endl;

	return true;
}

bool CAlgModuleMng::ProcessData()
{
	cout << "This is ProcessData..." << endl;

	return true;
}

CAlgModuleMng::CAlgModuleMng()
{
}

CAlgModuleMng::~CAlgModuleMng()
{
}

bool CAlgModuleMng::InitAlgModule(const string& strModuleType, const string& strModuleName, nlohmann::json&& cfgObj)
{
	DllShrPtr dllParser = DllShrPtr(new DllParser);

	//开始加载各个模型InitModuleDll处理函数
	if (dllParser->Load(strModuleName))
	{
		cout << "strModuleNmae" << strModuleName << endl;

		using MsgBusShrPtr = std::shared_ptr<common_messagebus::MessageBus>;
		using TupleType = std::tuple<MsgBusShrPtr, Json, char*>;
		TupleType&& Tuple = std::make_tuple(std::forward<MsgBusShrPtr>(SCMsgBusMng.GetMsgBus()), cfgObj,(char*)strModuleName.c_str());
		dllParser->ExcecuteFunc<bool(Any&&)>("InitModuleDll", std::move(Tuple));

		m_mapDllShrPtr.insert(make_pair(strModuleType + strModuleName, dllParser));
		LOG_INFO("systerm") << string_format("load module successful %s\n",strModuleName.c_str());

		if (strModuleName=="./libSlowFast.so")
		{
			dllParser->ExcecuteFunc<bool(Any&&)>("ProcSlowFast", std::move(Tuple));
			cout << "ProcSlowFast function is done !" << endl;
		}
	}
	else
	{
		LOG_INFO("systerm") << string_format("load module fail %s\n", strModuleName.c_str());
	}
	return true;
}
