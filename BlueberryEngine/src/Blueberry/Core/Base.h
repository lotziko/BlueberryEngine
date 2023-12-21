#pragma once

#include <wrl\client.h>
#include <memory>

namespace Blueberry
{
	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;
}

#undef min
#undef max

#define MAX_COMPONENTS 128
#define INVALID_ID -1
#define TO_STRING( x ) #x