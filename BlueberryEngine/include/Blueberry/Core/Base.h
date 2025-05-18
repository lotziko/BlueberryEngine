#pragma once

#ifdef BUILD_DLL
#define BB_API __declspec(dllimport)
#else
#define BB_API __declspec(dllexport)
#endif

#pragma warning(disable: 4251) // Hide STL dllexport warnings

#include "Blueberry\Core\Memory.h"
#include "Blueberry\Math\Math.h"
#include "Blueberry\Logging\Log.h"

#include <stdint.h>
#include <string>

#undef min
#undef max
#undef GetObject

#define INVALID_ID -1
#define TO_STRING( x ) #x
#define TO_HASH( x ) std::hash<String>()(x)