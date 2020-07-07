#include "IDeepSortExp.h"
#include "InitMng.h"

extern "C" bool InitModuleDll(common_template::Any&& anyObj)
{
	return common_template::CSingleton<DEEPSORT_ALG::CInitMng>::GetInstance().Init(std::move(anyObj));
}