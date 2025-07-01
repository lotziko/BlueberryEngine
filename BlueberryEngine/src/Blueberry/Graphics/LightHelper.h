#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Light;
	class Transform;

	struct LightRenderingData
	{
		Vector4 lightParam;
		Vector4 lightPosition;
		Vector4 lightColor;
		Vector4 lightAttenuation;
		Vector4 lightDirection;
	};

	class LightHelper
	{
	public:
		static Matrix GetViewMatrix(Light* light, Transform* transform, const uint8_t& slice = 0);
		static Matrix GetInverseViewMatrix(Light* light, Transform* transform, const uint8_t& slice = 0);
		static Matrix GetProjectionMatrix(Light* light, const float& guardAngle = 0);
		static void GetRenderingData(Light* light, Transform* transform, LightRenderingData& data);
	};
}