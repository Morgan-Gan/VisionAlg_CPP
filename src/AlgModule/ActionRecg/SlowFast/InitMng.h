#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include "tuple/TpIndexs.h"
#include "TpApply.h"
#include "Singleton.h"
#include "nlohmann/json.hpp"
#include "MessageBus.h"
#include "Any.h"
#include "TorchSource.h"
#include "ActionRecg.h"

namespace SLOWFAST_ALG
{
	class CInitMng : public common_template::CSingleton<CInitMng>
	{
		friend class common_template::CSingleton<CInitMng>;

		using Json = nlohmann::json;
		using MsgBusShrPtr = std::shared_ptr<common_messagebus::MessageBus>;
		using TupleType = std::tuple<MsgBusShrPtr, Json, char*>;

	public:
		bool Init(common_template::Any&& anyObj);
		bool InitSF(common_template::Any&& anyObj);

	private:
		CInitMng();
		~CInitMng();

		bool InitParams_SF(MsgBusShrPtr ptrMegBus, Json cfgObj, char* pData);
		bool ProcMat_SF(TorchTensor& output, MatVec& vecBuf, FloatVec& vecScale);

	private:
		//定义另一个类的指针，需要在主类里面初始化指针
		CActionRecg* m_pActionRecg;
		MsgBusShrPtr m_ptrMsgBus;

	};
#define SCInitMng (common_template::CSingleton<CInitMng>::GetInstance())
}