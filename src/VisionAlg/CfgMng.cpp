#include <iostream>
#include <fstream>
#include <sstream>
#include "CfgMng.h"

using namespace std;
using namespace VISION_ALG;
using namespace common_commonobj;

void CCfgMng::ReadJsonFile(const string& strPath, string& strJson)
{
	//从文件中读取
	ifstream fin(strPath.c_str(), ios::binary);

	//创建字符串流对象
	ostringstream sin;

	//把文件流中的字符输入到字符串流中
	sin << fin.rdbuf();

	//获取字符串流中的字符串
	strJson = sin.str();

	//关闭和清除文件流对象
	fin.close();
	fin.clear();
}

//加载json配置
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