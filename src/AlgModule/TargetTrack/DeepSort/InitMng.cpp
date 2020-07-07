#include "InitMng.h"
#include "log4cxx/Loging.h"
#include "BoostFun.h"

using namespace std;
using namespace common_commonobj;
using namespace DEEPSORT_ALG;

CInitMng::CInitMng()
{
}

CInitMng::~CInitMng()
{
}

bool CInitMng::InitParams(MsgBusShrPtr ptrMegBus, Json cfgObj, char* pData)
{
	LOG_INFO("systerm") << string_format("Init dll successful %s", pData);
	return true;
}

bool CInitMng::Init(common_template::Any&& anyObj)
{
	return common_tuple::apply(std::bind(&CInitMng::InitParams,this,placeholders::_1, placeholders::_2,placeholders::_3),anyObj.AnyCast<TupleType>()) ;
}