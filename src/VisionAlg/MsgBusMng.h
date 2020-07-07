#pragma once
#include <memory>
#include "Singleton.h"
#include "MessageBus.h"

namespace VISION_ALG
{
	class CMsgBusMng : public common_template::CSingleton<CMsgBusMng>
	{
		friend class common_template::CSingleton<CMsgBusMng>;
		//��������ʹ�� using ������������Ա����,���Ա��������
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
