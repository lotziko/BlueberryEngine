#pragma once

#include <wrl\client.h>
#include <memory>

namespace Blueberry
{
	template<typename T>
	using ComPtr = Microsoft::WRL::ComPtr<T>;

	template<typename T>
	using SharedPtr = std::shared_ptr<T>;
	template<typename T, typename ... Args>
	constexpr SharedPtr<T> CreateSharedPtr(Args&& ... args)
	{
		return std::make_shared<T>(std::forward<Args>(args)...);
	}

	template<typename T>
	using UniquePtr = std::unique_ptr<T>;
	template<typename T, typename ... Args>
	constexpr UniquePtr<T> CreateUniquePtr(Args&& ... args)
	{
		return std::make_unique<T>(std::forward<Args>(args)...);
	}

	using byte = unsigned char;
}

#undef min
#undef max
#undef GetObject

#define MAX_COMPONENTS 128
#define INVALID_ID -1
#define TO_STRING( x ) #x
#define TO_HASH( x ) std::hash<std::string>()(x)