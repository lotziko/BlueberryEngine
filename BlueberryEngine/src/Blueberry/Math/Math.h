#pragma once

#include "SimpleMath.h"

namespace Blueberry
{
	using Vector2 = DirectX::SimpleMath::Vector2;
	using Vector3 = DirectX::SimpleMath::Vector3;
	using Vector4 = DirectX::SimpleMath::Vector4;
	using Quaternion = DirectX::SimpleMath::Quaternion;
	using Ray = DirectX::SimpleMath::Ray;
	using Matrix = DirectX::SimpleMath::Matrix;
	using Color = DirectX::SimpleMath::Color;
	using Rectangle = DirectX::SimpleMath::Rectangle;
	using Viewport = DirectX::SimpleMath::Viewport;

	constexpr auto Pi = 3.1415926535f;
	constexpr auto DegreeToRad = 57.29577951471995f;
	constexpr auto RadToDegree = 0.0174532925194444f;

	inline float ToDegrees(float radians)
	{
		return radians * DegreeToRad;
	}

	inline Vector3 ToDegrees(Vector3 radians)
	{
		return Vector3(radians.x * DegreeToRad, radians.y * DegreeToRad, radians.z * DegreeToRad);
	}

	inline float ToRadians(float degrees)
	{
		return degrees * RadToDegree;
	}

	inline Vector3 ToRadians(Vector3 degrees)
	{
		return Vector3(degrees.x * RadToDegree, degrees.y * RadToDegree, degrees.z * RadToDegree);
	}
}