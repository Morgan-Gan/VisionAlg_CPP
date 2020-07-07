#include <iostream>
#include <fstream>
#include <sstream>
#include "CfgMng.h"

using namespace std;
using namespace VISION_ALG;
using namespace common_commonobj;

void CCfgMng::ReadJsonFile(const string& strPath, string& strJson)
{
	//���ļ��ж�ȡ
	ifstream fin(strPath.c_str(), ios::binary);

	//�����ַ���������
	ostringstream sin;

	//���ļ����е��ַ����뵽�ַ�������
	sin << fin.rdbuf();

	//��ȡ�ַ������е��ַ���
	strJson = sin.str();

	//�رպ�����ļ�������
	fin.close();
	fin.clear();
}

//����json����
bool CCfgMng::LoadJsonCfg(const string& strPath)
{
	string strJson;
	ReadJsonFile(strPath, strJson);

	if (strJson.empty())
	{
		return std::false_type::value;
	}

	m_jsObj = nlohmann::json::parse(strJson.c_str());

	return std::true_type::value;
}

bool CCfgMng::LoadCfg()
{
	string strPath("./config/config.json");

	return LoadJsonCfg(strPath);
}

Json CCfgMng::GetJsonCfg()
{
	return m_jsObj;
}

CCfgMng::CCfgMng()
{
}

CCfgMng::~CCfgMng()
{
}