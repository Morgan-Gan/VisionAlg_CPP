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
	//���ü��س�ʼ��
	if (!SCCfgMng.LoadCfg())
	{
		return false;
	}

	//�����㷨ģ��
	if (!SCAlgModuleMng.LoadAlgModule())
	{
		return 0;
	}

	//�㷨����
	if (!SCAlgModuleMng.OperateAlgModule())
	{
		return 0;
	}
}


bool ProcData(cv::Mat data, torch::Tensor imgtensor, torch::Tensor bbox)
{
	//��������
	if (!SCAlgModuleMng.LoadData(data, imgtensor))
	{
		return 0;
	}

	//��������
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
	//����anyObj���������κβ�������ת��Ϊtuple
	using TupleType = std::tuple<cv::Mat, torch::Tensor, torch::Tensor>;
	//ռλ��_1��˳����Ե���
	return common_tuple::apply(std::bind(ProcData, placeholders::_1, placeholders::_2, placeholders::_3), anyObj.AnyCast<TupleType>());
}

