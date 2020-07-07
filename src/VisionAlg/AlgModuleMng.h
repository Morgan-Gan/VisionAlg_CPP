#pragma once
#include <unordered_map>
#include "CommonDataType.h"
#include "TorchSource.h"


namespace VISION_ALG
{
	class CAlgModuleMng : public common_template::CSingleton<CAlgModuleMng>
	{
		friend class common_template::CSingleton<CAlgModuleMng>;
		
	public:
		bool LoadAlgModule();
		bool OperateAlgModule();

		bool LoadData(cv::Mat data, torch::Tensor imgtensor);
		bool ProcessData();

	private:
		CAlgModuleMng();
		~CAlgModuleMng();

		bool InitAlgModule(const std::string& strModuleType,const std::string& strModuleName, Json&& cfgObj);


	private:
		std::unordered_map<std::string, DllShrPtr> m_mapDllShrPtr;
	};
#define  SCAlgModuleMng (common_template::CSingleton<CAlgModuleMng>::GetInstance())
}