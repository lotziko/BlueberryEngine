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
}