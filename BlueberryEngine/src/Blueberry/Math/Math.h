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

	inline float Max(float a, float b)
	{
		return a > b ? a : b;
	}

	inline float Min(float a, float b)
	{
		return a < b ? a : b;
	}

	inline Vector3 MultiplyPoint(Matrix matrix, Vector3 point)
	{
		Vector4 pointToTransform = Vector4(point.x, point.y, point.z, 1.0f);
		pointToTransform = Vector4::Transform(pointToTransform, matrix);
		pointToTransform.w = 1 / pointToTransform.w;
		pointToTransform.x *= pointToTransform.w;
		pointToTransform.y *= pointToTransform.w;
		pointToTransform.z *= pointToTransform.w;
		return Vector3(pointToTransform.x, pointToTransform.y, pointToTransform.z);
	}

	inline Vector3 MultiplyVector(Matrix matrix, Vector3 vector)
	{
		Vector4 vectorToTransform = Vector4(vector.x, vector.y, vector.z, 0.0f);
		vectorToTransform = Vector4::Transform(vectorToTransform, matrix);
		return Vector3(vectorToTransform.x, vectorToTransform.y, vectorToTransform.z);
	}
}