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
		static uint32_t GetShadowSize(LightType type);
		static float GetShadowSlopeBias(LightType type, uint8_t slice);
		static uint32_t GetSliceCount(LightType type);
		static Matrix GetViewMatrix(Light* light, Transform* transform, uint8_t slice = 0);
		static Matrix GetInverseViewMatrix(Light* light, Transform* transform, uint8_t slice = 0);
		static Matrix GetProjectionMatrix(Light* light, float guardAngle = 0);
		static Vector4 GetAttenuation(LightType type, float lightRange, float spotOuterAngle, float spotInnerAngle);
	};
}