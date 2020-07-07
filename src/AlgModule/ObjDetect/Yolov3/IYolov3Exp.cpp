#include "IYolov3Exp.h"
#include "ModuleMng.h"

extern "C" bool InitModuleDll(common_template::Any&& anyObj)
{
	return common_template::CSingleton<YOLOV3_ALG::CModuleMng>::GetInstance().Init(std::move(anyObj));
}