#pragma once

#include <wrl\client.h>
#include <memory>

namespace Blueberry
{
	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;

	using byte = unsigned char;
}

#undef min
#undef max
#undef GetObject

#define MAX_COMPONENTS 128
#define INVALID_ID -1
#define TO_STRING( x ) #x
#define TO_HASH( x ) std::hash<std::string>()(x)