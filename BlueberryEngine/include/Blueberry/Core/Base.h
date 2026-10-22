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

	template <typename T>
	using UniquePtr = std::unique_ptr<T>;
	template <typename T, typename ... Args>
	constexpr UniquePtr<T> MakeUnique(Args&& ... args)
	{
		return std::make_unique<T>(std::forward<Args>(args)...);
	}

	template <typename T>
	using SharedPtr = std::shared_ptr<T>;
	template<typename T, typename ... Args>
	constexpr SharedPtr<T> MakeShared(Args&& ... args)
	{
		return std::make_shared<T>(std::forward<Args>(args)...);
	}
}