#pragma once

#include <wrl\client.h>
#include <memory>

namespace Blueberry
{
	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;
}