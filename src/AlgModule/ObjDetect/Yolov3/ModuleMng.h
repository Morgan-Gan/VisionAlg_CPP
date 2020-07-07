#pragma once
#include "CommonDataType.h"
#include "ObjDetect.h"
#include "SubTopic.h"

namespace YOLOV3_ALG
{
	class CModuleMng : public common_template::CSingleton<CModuleMng>
	{
		friend class common_template::CSingleton<CModuleMng>;

	public:
		bool Init(common_template::Any&& anyObj);

	private:
		CModuleMng();
		~CModuleMng();

		bool InitParams(MsgBusShrPtr ptrMegBus, Json cfgObj, char* pData);
		bool ProcMat(const Mat& srcMat, TorchTensor& output, MatVec& vecBuf, FloatVec& vecScale);

	private:
		CObjDetect* m_pObjDetect;
		MsgBusShrPtr m_ptrMsgBus;
	};
#define SCInitMng (common_template::CSingleton<CModuleMng>::GetInstance())
}