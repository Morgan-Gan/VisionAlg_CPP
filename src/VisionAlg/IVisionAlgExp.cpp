#include "IVisionAlgExp.h"
#include "TpApply.h"
#include "log4cxx/Loging.h"
#include "CfgMng.h"
#include "AlgModuleMng.h"
#include <string>

using namespace std;
using namespace VISION_ALG;
using namespace common_commonobj;

bool InitIFParse()
{
	//配置加载初始化
	if (!SCCfgMng.LoadCfg())
	{
		return false;
	}

	//加载算法模块
	if (!SCAlgModuleMng.LoadAlgModule())
	{
		return 0;
	}

	//算法操作
	if (!SCAlgModuleMng.OperateAlgModule())
	{
		return 0;
	}
}


bool ProcData(cv::Mat data, torch::Tensor imgtensor, torch::Tensor bbox)
{
	//加载数据
	if (!SCAlgModuleMng.LoadData(data, imgtensor))
	{
		return 0;
	}

	//处理数据
	if (!SCAlgModuleMng.ProcessData())
	{
		return 0;
	}
}

IVISIONALG_API bool InitModuleDll(common_template::Any&& anyObj)
{
	using TupleType = std::tuple<>;
	return common_tuple::apply(InitIFParse, anyObj.AnyCast<TupleType>());
}

IVISIONALG_API bool ProcModuleDll(common_template::Any&& anyObj)
{
	//解析anyObj传过来的任何参数，并转化为tuple
	using TupleType = std::tuple<cv::Mat, torch::Tensor, torch::Tensor>;
	//占位符_1，顺序可以调换
	return common_tuple::apply(std::bind(ProcData, placeholders::_1, placeholders::_2, placeholders::_3), anyObj.AnyCast<TupleType>());
}

