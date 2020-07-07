#pragma once
#include <memory>
#include "Singleton.h"
#include "MessageBus.h"

namespace VISION_ALG
{
	class CMsgBusMng : public common_template::CSingleton<CMsgBusMng>
	{
		friend class common_template::CSingleton<CMsgBusMng>;
		//在子类中使用 using 声明引入基类成员名称,可以被子类访问
		using MsgBusShrPtr = std::shared_ptr<common_messagebus::MessageBus>;
	public:
		MsgBusShrPtr GetMsgBus();

	private:
		CMsgBusMng();
		~CMsgBusMng();

	private:
		MsgBusShrPtr m_ptrMessageBus;
	};

#define  SCMsgBusMng (common_template::CSingleton<CMsgBusMng>::GetInstance())
}
