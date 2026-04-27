#pragma once

#ifdef BUILD_DLL
#define BB_API __declspec(dllimport)
#else
#define BB_API __declspec(dllexport)
#endif

#pragma warning(disable: 4251) // Hide STL dllexport warnings
#pragma warning(disable: 26812) // Hide unscoped enum warnings

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

namespace Blueberry
{
	using TypeId = uint32_t;
	using ObjectId = int32_t;
}