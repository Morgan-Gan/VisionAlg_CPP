#include "ModuleMng.h"
#include "BoostFun.h"

using namespace std;
using namespace common_commonobj;
using namespace YOLOV3_ALG;

CModuleMng::CModuleMng() : m_pObjDetect(nullptr)
{
}

CModuleMng::~CModuleMng()
{
}

bool CModuleMng::InitParams(MsgBusShrPtr ptrMegBus, Json cfgObj, char* pData)
{
	//初始化目标探测
	if (!m_pObjDetect)
	{
		m_pObjDetect = new CObjDetect;
	}
	m_pObjDetect->DetectInit();

	//消息订阅
	m_ptrMsgBus = ptrMegBus;

	std::string&& strTopic = std::string("Yolov3ProcMatExt");
	m_ptrMsgBus->Attach([this](const Mat& srcMat, TorchTensor& output, MatVec& vecBuf, FloatVec& vecScale)->bool {return ProcMat(srcMat, output, vecBuf,vecScale); }, strTopic);

	LOG_INFO("systerm") << string_format("Init dll successful %s", pData);
	return true;
}

bool CModuleMng::ProcMat(const Mat& srcMat, TorchTensor& output, MatVec& vecBuf, FloatVec& vecScale)
{
	if (m_pObjDetect)
	{
		m_pObjDetect->DetectOp(srcMat, output, vecBuf, vecScale);
	}
	return true;
}


bool CModuleMng::Init(common_template::Any&& anyObj)
{
	using TupleTypeInit = std::tuple<MsgBusShrPtr, Json, char*>;
	return common_tuple::apply(std::bind(&CModuleMng::InitParams, this, placeholders::_1, placeholders::_2, placeholders::_3), anyObj.AnyCast<TupleTypeInit>());
}