#include "LightHelper.h"

#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	static Matrix s_PointLightMatrices[6] =
	{
		Matrix::CreateLookAt(Vector3::Zero, Vector3::Right, Vector3::Up),
		Matrix::CreateLookAt(Vector3::Zero, Vector3::Left, Vector3::Up),
		Matrix::CreateLookAt(Vector3::Zero, Vector3::Up, Vector3::Backward),
		Matrix::CreateLookAt(Vector3::Zero, Vector3::Down, Vector3::Forward),
		Matrix::CreateLookAt(Vector3::Zero, Vector3::Forward, Vector3::Up),
		Matrix::CreateLookAt(Vector3::Zero, Vector3::Backward, Vector3::Up)
	};

	static Vector3 s_PointLightDirection[6] =
	{
		Vector3::Right, Vector3::Left, Vector3::Up, Vector3::Down, Vector3::Forward, Vector3::Backward
	};

	static Vector3 s_PointLightUp[6] =
	{
		Vector3::Up, Vector3::Up, Vector3::Backward, Vector3::Forward, Vector3::Up, Vector3::Up
	};

	Matrix LightHelper::GetViewMatrix(Light* light, Transform* transform, const uint8_t& slice)
	{
		switch (light->GetType())
		{
		case LightType::Spot:
		case LightType::Directional:
			return transform->GetWorldToLocalMatrix();
		case LightType::Point:
		{
			Vector3 position = transform->GetPosition();
			return Matrix::CreateLookAt(position, position + s_PointLightDirection[slice], s_PointLightUp[slice]); // Something broken here
		}
		default:
			return Matrix::Identity;
		}
	}

	Matrix LightHelper::GetInverseViewMatrix(Light* light, Transform* transform, const uint8_t& slice)
	{
		switch (light->GetType())
		{
		case LightType::Spot:
		case LightType::Directional:
			return transform->GetLocalToWorldMatrix();
		case LightType::Point:
			return s_PointLightMatrices[slice] * transform->GetLocalToWorldMatrix(); // Something broken here
		default:
			return Matrix::Identity;
		}
	}

	Matrix LightHelper::GetProjectionMatrix(Light* light, const float& guardAngle)
	{
		switch (light->GetType())
		{
		case LightType::Spot:
			return Matrix::CreatePerspectiveFieldOfView(ToRadians(light->GetOuterSpotAngle()), 1, 0.01f, light->GetRange());
		case LightType::Point:
			return Matrix::CreatePerspectiveFieldOfView(ToRadians(90 + guardAngle), 1, 0.01f, light->GetRange());
		default:
			return Matrix::Identity;
		}
	}

	Vector4 LightHelper::GetAttenuation(LightType type, float lightRange, float spotOuterAngle, float spotInnerAngle)
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
}