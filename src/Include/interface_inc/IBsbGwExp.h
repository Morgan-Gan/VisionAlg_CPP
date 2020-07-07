#pragma once
#include <string>
#include <iostream>
#include "Any.h"

#ifdef __linux
#define IBSBGW_API extern "C"
#else
#ifdef IBSBGW_EXPORT
#define IBSBGW_API extern "C" __declspec(dllexport)
#else
#define IBSBGW_API extern "C" __declspec(dllimport)
#endif
#endif

IBSBGW_API bool InitModuleDll(common_template::Any& anyObj);
IBSBGW_API bool InitBsbGwModuleDll(common_template::Any& anyObj);
