#pragma once

#include <wrl\client.h>
#include <memory>

namespace Blueberry
{
	template<typename T>
	using ComRef = Microsoft::WRL::ComPtr<T>;

	template<typename T>
	using Ref = std::shared_ptr<T>;
	template<typename T, typename ... Args>
	constexpr Ref<T> CreateRef(Args&& ... args)
	{
		return std::make_shared<T>(std::forward<Args>(args)...);
	}

	template<typename T>
	using Scope = std::unique_ptr<T>;
	template<typename T, typename ... Args>
	constexpr Scope<T> CreateScope(Args&& ... args)
	{
		return std::make_unique<T>(std::forward<Args>(args)...);
	}
}

#define MAX_COMPONENTS 128
#define INVALID_ID -1
#define TO_STRING( x ) #x