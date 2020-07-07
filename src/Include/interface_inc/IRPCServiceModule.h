/*
 * =====================================================================================
 *
 *       Filename:  IRPCServiceModule.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/20/2019 03:26:42 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *        Company:
 *
 * =====================================================================================
 */
 /*************************************************************************
	 > File Name: ICTCPServiceModule.h
	 > Author: ma6174
	 > Mail: ma6174@163.com
	 > Created Time: Tue 20 Oct 2015 03:26:42 PM HKT
  ************************************************************************/
#ifndef BSB_IRPCSERVICEMODULE_H_
#define BSB_IRPCSERVICEMODULE_H_
#include <string>
#include <functional>
#include "CommonFun.h"
#include "Any.h"

namespace transmission_rpc
{
	using Func = std::function<void()>;
	extern "C" bool GetRpc(common_template::Any&& RpcClient, common_template::Any&& RpcService);
	extern "C" bool CallRpcFunc(Func&& func, const std::string& strFuncName, common_template::Any&& AnyClientObj);
	extern "C" void RpcInit(const char* pAppIp, int nApport, int nMaxRpcConn);
}
#endif