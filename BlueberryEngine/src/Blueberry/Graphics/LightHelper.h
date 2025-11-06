#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Light;
	class Transform;

	enum class LightType;

	class LightHelper
	{
	public:
		static Matrix GetViewMatrix(Light* light, Transform* transform, const uint8_t& slice = 0);
		static Matrix GetInverseViewMatrix(Light* light, Transform* transform, const uint8_t& slice = 0);
		static Matrix GetProjectionMatrix(Light* light, const float& guardAngle = 0);
		static Vector4 GetAttenuation(LightType type, float lightRange, float spotOuterAngle, float spotInnerAngle);
	};
}