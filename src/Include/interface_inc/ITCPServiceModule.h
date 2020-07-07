#ifndef BSB_ITCPSERVICEMODULE_H_
#define BSB_ITCPSERVICEMODULE_H_
#include <string>
#include <vector>
#include <tuple>

class ICTCPServiceCallback
{
public:
	virtual void OnTcpConnect(const std::string& strPeerConnKey, const std::string& strLocalConnKey, bool bStatus) = 0;
	virtual void OnTcpMessage(const std::string& strConnKey, const std::string& strLocalConnKey, const char* pData, const int nDataLen) = 0;
};

class ICTCPServiceModule
{
public:
	virtual bool StartTCPServer(std::vector<std::tuple<unsigned short,int>>&& vecServerCfg) = 0;
	virtual void SetCallbackObj(ICTCPServiceCallback* pCallbackObj) = 0;
	virtual bool SendData(const std::string& strConnKey, const std::string& strData, const int& s32TcpServerPort) = 0;
	virtual void CloseConnect(const std::string& strConnKey, const int& s32TcpServerPort) = 0;
};

extern "C" ICTCPServiceModule* GetTCPServiceModuleInstance();
#endif
