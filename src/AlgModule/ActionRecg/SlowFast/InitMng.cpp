#include "InitMng.h"
#include "log4cxx/Loging.h"
#include "BoostFun.h"
#include "ActionRecg.h"


using namespace std;
using namespace common_commonobj;
using namespace SLOWFAST_ALG;

#define MODEL_PATH ("./model/")

CInitMng::CInitMng() : m_pActionRecg(nullptr)                    
{
}

CInitMng::~CInitMng()
{
}

bool CInitMng::InitParams_SF(MsgBusShrPtr ptrMegBus, Json cfgObj, char* pData)
{
	if (!pData)
	{
		LOG_DEBUG("slowfast_alg") << "pData is null\n";
		return false;
	}

	//初始化动作识别,new存堆中，不用new存栈中
	if (!m_pActionRecg)
	{
		m_pActionRecg = new CActionRecg;
	}
	m_pActionRecg->RecgInit();

	//向某个主题注册主题，需要订阅主题（topic、消息类型）和消息出题函数
	m_ptrMsgBus = ptrMegBus;
	std::string&& strTopic = std::string("SlowFastProcMatExt");
	m_ptrMsgBus->Attach([this](TorchTensor& output, MatVec& vecBuf, FloatVec& vecScale)->bool {return ProcMat_SF(output, vecBuf, vecScale); }, strTopic);

	LOG_INFO("slowfast_alg") << string_format("Init dll successful %s", pData);


	return true;
}

bool CInitMng::Init(common_template::Any&& anyObj)
{
	//CActionRecg Op;
	//Op.RecgInit();
	//Op.RecgOp();

	using TupleTypeInit = std::tuple<MsgBusShrPtr, Json, char*>;
	return common_tuple::apply(std::bind(&CInitMng::InitParams_SF, this, placeholders::_1, placeholders::_2, placeholders::_3), anyObj.AnyCast<TupleTypeInit>());
}

bool CInitMng::ProcMat_SF(TorchTensor& output, MatVec& vecBuf, FloatVec& vecScale)
{
	if (m_pActionRecg)
	{
		m_pActionRecg->Test(output, vecBuf, vecScale);
	}
	return true;
}

bool CInitMng::InitSF(common_template::Any&& anyObj)
{
	//if (m_pActionRecg)
	//{
	//	m_pActionRecg->Load_data();
	//}
	cout << "For information pass..." << endl;

	return true;

}