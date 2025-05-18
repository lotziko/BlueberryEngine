#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Light;

	class LightHelper
	{
	public:
		static Matrix GetViewMatrix(Light* light, const uint8_t& slice = 0);
		static Matrix GetProjectionMatrix(Light* light, const uint8_t& slice = 0);
	};
}