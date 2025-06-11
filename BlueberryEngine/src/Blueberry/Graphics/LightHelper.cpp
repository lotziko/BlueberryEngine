#include "LightHelper.h"

#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	Matrix LightHelper::GetViewMatrix(Light* light, const uint8_t& slice)
	{
		switch (light->GetType())
		{
		case LightType::Spot:
		case LightType::Directional:
			return light->GetTransform()->GetWorldToLocalMatrix();
		default:
			return Matrix::Identity;
		}
	}

	Matrix LightHelper::GetProjectionMatrix(Light* light, const uint8_t& slice)
	{
		switch (light->GetType())
		{
		case LightType::Spot:
			return Matrix::CreatePerspectiveFieldOfView(ToRadians(light->GetOuterSpotAngle()), 1, 0.01f, light->GetRange());
		default:
			return Matrix::Identity;
		}
	}

	Vector4 GetAttenuation(LightType type, float lightRange, float spotOuterAngle, float spotInnerAngle)
	{
		Vector4 lightAttenuation = Vector4(0.0f, 1.0f, 0.0f, 1.0f);

		if (type != LightType::Directional)
		{
			float lightRangeSqr = lightRange * lightRange;
			float fadeStartDistanceSqr = 0.8f * 0.8f * lightRangeSqr;
			float fadeRangeSqr = (fadeStartDistanceSqr - lightRangeSqr);
			float lightRangeSqrOverFadeRangeSqr = -lightRangeSqr / fadeRangeSqr;
			float oneOverLightRangeSqr = 1.0f / std::max(0.0001f, lightRangeSqr);

			lightAttenuation.x = oneOverLightRangeSqr;
			lightAttenuation.y = lightRangeSqrOverFadeRangeSqr;
		}

		if (type == LightType::Spot)
		{
			float cosOuterAngle = cos(ToRadians(spotOuterAngle) * 0.5f);
			float cosInnerAngle = cos(ToRadians(spotInnerAngle) * 0.5f);
			float smoothAngleRange = std::max(0.001f, cosInnerAngle - cosOuterAngle);
			float invAngleRange = 1.0f / smoothAngleRange;
			float add = -cosOuterAngle * invAngleRange;

			lightAttenuation.z = invAngleRange;
			lightAttenuation.w = add;
		}

		return lightAttenuation;
	}

	void LightHelper::GetRenderingData(Light* light, Transform* transform, LightRenderingData& data)
	{
		Vector4 position;
		Vector4 direction = Vector4(0.0f, 0.0f, 1.0f, 0.0f);
		LightType type = light->GetType();
		Color color = light->GetColor();
		float intensity = light->GetIntensity();
		float range = light->GetRange();
		float outerSpotAngle = light->GetOuterSpotAngle();
		float innerSpotAngle = light->GetInnerSpotAngle();

		if (type == LightType::Spot)
		{
			Vector3 dir = Vector3::Transform(Vector3::Backward, transform->GetRotation());
			direction = Vector4(dir.x, dir.y, dir.z, 0.0f);
		}

		if (type == LightType::Directional)
		{
			Vector3 dir = Vector3::Transform(Vector3::Backward, transform->GetRotation());
			position = Vector4(dir.x, dir.y, dir.z, 0.0f);
		}
		else
		{
			Vector3 pos = transform->GetPosition();
			position = Vector4(pos.x, pos.y, pos.z, 1.0f);
		}

		data.lightParam = Vector4(light->IsCastingShadows() ? 1.0f : 0.0f, light->GetCookie() == nullptr ? 0.0f : 1.0f, 0.0f, range * range);
		data.lightPosition = position;
		data.lightColor = Vector4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
		data.lightAttenuation = GetAttenuation(type, range, outerSpotAngle, innerSpotAngle);
		data.lightDirection = direction;
	}
}