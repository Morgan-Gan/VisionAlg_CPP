#pragma once
#include "CommonDataType.h"

namespace VISION_ALG
{
	class CCfgMng : public common_template::CSingleton<CCfgMng>
	{
		friend class common_template::CSingleton<CCfgMng>;
	public:
		bool LoadCfg();
		Json GetJsonCfg();

	private:
		CCfgMng();
		~CCfgMng();

	private:
		bool LoadJsonCfg(const std::string& strPath);
		void ReadJsonFile(const std::string& strPath, std::string& strJson);

	private:
		Json m_jsObj;
	};

#define SCCfgMng (common_template::CSingleton<CCfgMng>::GetInstance())
}