#include "MsgBusMng.h"

using namespace VISION_ALG;
using namespace common_messagebus;

CMsgBusMng::CMsgBusMng()
{
	m_ptrMessageBus.reset(new MessageBus);
}

CMsgBusMng::~CMsgBusMng()
{
}

CMsgBusMng::MsgBusShrPtr CMsgBusMng::GetMsgBus()
{
	return m_ptrMessageBus;
}
