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

namespace DEEPSORT_ALG
{
	class CInitMng : public common_template::CSingleton<CInitMng>
	{
		friend class common_template::CSingleton<CInitMng>;
		
		using Json = nlohmann::json;
		using MsgBusShrPtr = std::shared_ptr<common_messagebus::MessageBus>;
		using TupleType = std::tuple<MsgBusShrPtr, Json, char*>;

	public:
		bool Init(common_template::Any&& anyObj);
	
	private:
		CInitMng();
		~CInitMng();

		bool InitParams(MsgBusShrPtr ptrMegBus, Json cfgObj, char* pData);
	};
#define SCInitMng (common_template::CSingleton<CInitMng>::GetInstance())
}