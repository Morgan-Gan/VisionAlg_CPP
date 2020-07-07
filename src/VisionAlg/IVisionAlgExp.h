#pragma once
#include <string>
#include <iostream>
#include "Any.h"
#include "TorchSource.h"

//using namespace std;
//using namespace cv;

#ifdef __linux
#define IVISIONALG_API extern "C"
#else
#ifdef IVISIONALG_EXPORT
#define IVISIONALG_API extern "C" __declspec(dllexport)
#else
#define IVISIONALG_API extern "C" __declspec(dllimport)
#endif
#endif

IVISIONALG_API bool InitModuleDll(common_template::Any&& anyObj);

IVISIONALG_API bool ProcModuleDll(common_template::Any&& anyObj);