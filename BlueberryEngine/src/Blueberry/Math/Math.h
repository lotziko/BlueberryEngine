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
	using AABB = DirectX::BoundingBox;
	using Frustum = DirectX::BoundingFrustum;

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

	inline int Max(int a, int b)
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

	inline DirectX::ContainmentType ContainsFlipped(const DirectX::BoundingFrustum& frustum, const DirectX::BoundingBox& box)
	{
		using XMVECTOR = DirectX::XMVECTOR;
		// Load origin and orientation of the frustum.
		XMVECTOR vOrigin = XMLoadFloat3(&frustum.Origin);
		XMVECTOR vOrientation = XMLoadFloat4(&frustum.Orientation);

		// Create 6 planes (do it inline to encourage use of registers)
		XMVECTOR NearPlane = DirectX::XMVectorSet(0.0f, 0.0f, 1.0f, frustum.Near); // Flipped
		NearPlane = DirectX::Internal::XMPlaneTransform(NearPlane, vOrientation, vOrigin);
		NearPlane = DirectX::XMPlaneNormalize(NearPlane);

		XMVECTOR FarPlane = DirectX::XMVectorSet(0.0f, 0.0f, 1.0f, frustum.Far); // Flipped
		FarPlane = DirectX::Internal::XMPlaneTransform(FarPlane, vOrientation, vOrigin);
		FarPlane = DirectX::XMPlaneNormalize(FarPlane);

		XMVECTOR RightPlane = DirectX::XMVectorSet(1.0f, 0.0f, -frustum.RightSlope, 0.0f);
		RightPlane = DirectX::Internal::XMPlaneTransform(RightPlane, vOrientation, vOrigin);
		RightPlane = DirectX::XMPlaneNormalize(RightPlane);

		XMVECTOR LeftPlane = DirectX::XMVectorSet(-1.0f, 0.0f, frustum.LeftSlope, 0.0f);
		LeftPlane = DirectX::Internal::XMPlaneTransform(LeftPlane, vOrientation, vOrigin);
		LeftPlane = DirectX::XMPlaneNormalize(LeftPlane);

		XMVECTOR TopPlane = DirectX::XMVectorSet(0.0f, 1.0f, -frustum.TopSlope, 0.0f);
		TopPlane = DirectX::Internal::XMPlaneTransform(TopPlane, vOrientation, vOrigin);
		TopPlane = DirectX::XMPlaneNormalize(TopPlane);

		XMVECTOR BottomPlane = DirectX::XMVectorSet(0.0f, -1.0f, frustum.BottomSlope, 0.0f);
		BottomPlane = DirectX::Internal::XMPlaneTransform(BottomPlane, vOrientation, vOrigin);
		BottomPlane = DirectX::XMPlaneNormalize(BottomPlane);

		return box.ContainedBy(NearPlane, FarPlane, RightPlane, LeftPlane, TopPlane, BottomPlane);
	}

	inline UINT GetMipCount(const UINT& width, const UINT& height, const bool& generateMips)
	{
		if (generateMips)
		{
			UINT mipCount = (UINT)log2(Max((float)width, (float)height));
			// Based on https://stackoverflow.com/questions/108318/how-can-i-test-whether-a-number-is-a-power-of-2
			if ((width & (width - 1)) == 0 && (height & (height - 1)) == 0)
			{
				return mipCount;
			}
		}
		return 1;
	}
}