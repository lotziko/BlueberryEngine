#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"

#include <functional>

namespace Blueberry
{
	class BB_API Timer
	{
	public:
		static size_t Start(float time, Object* object, const std::function<void()>& callback);
		static void Stop(size_t handle);
		static void Update();
	};
}