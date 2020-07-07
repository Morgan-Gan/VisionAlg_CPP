#include "ISlowFastExp.h"
#include "InitMng.h"

extern "C" bool InitModuleDll(common_template::Any&& anyObj)
{
	return common_template::CSingleton<SLOWFAST_ALG::CInitMng>::GetInstance().Init(std::move(anyObj));
}

extern "C" bool ProcSlowFast(common_template::Any&& anyObj)
{
	return common_template::CSingleton<SLOWFAST_ALG::CInitMng>::GetInstance().InitSF(std::move(anyObj));
}