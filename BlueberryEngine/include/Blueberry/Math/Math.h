#pragma once

#include "SimpleMath.h"

#include <algorithm>

namespace Blueberry
{
	#undef max
	#undef min

	using Vector2 = DirectX::SimpleMath::Vector2;
	using Vector3 = DirectX::SimpleMath::Vector3;
	using Vector4 = DirectX::SimpleMath::Vector4;
	using Vector2Int = DirectX::XMINT2;
	using Vector3Int = DirectX::XMINT3;
	using Vector4Int = DirectX::XMINT4;
	using Quaternion = DirectX::SimpleMath::Quaternion;
	using Ray = DirectX::SimpleMath::Ray;
	using Matrix = DirectX::SimpleMath::Matrix;
	using Color = DirectX::SimpleMath::Color;
	using Rectangle = DirectX::SimpleMath::Rectangle;
	using Viewport = DirectX::SimpleMath::Viewport;
	using AABB = DirectX::BoundingBox;
	using OBB = DirectX::BoundingOrientedBox;
	using Sphere = DirectX::BoundingSphere;
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

	inline Matrix CreateTRS(Vector3 position, Quaternion rotation, Vector3 scale)
	{
		return Matrix::CreateScale(scale) * Matrix::CreateFromQuaternion(rotation) * Matrix::CreateTranslation(position);
	}

	inline Quaternion LookRotation(Vector3 forward, Vector3 up = Vector3::UnitY)
	{
		Vector3 zAxis = forward;
		Vector3 xAxis = zAxis.Cross(up);
		xAxis.Normalize();
		Vector3 yAxis = zAxis.Cross(xAxis);

		Matrix rotationMatrix = Matrix(
			xAxis.x, xAxis.y, xAxis.z, 0.0f,
			yAxis.x, yAxis.y, yAxis.z, 0.0f,
			zAxis.x, zAxis.y, zAxis.z, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);

		return Quaternion::CreateFromRotationMatrix(rotationMatrix);
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

	inline void GetOrthographicPlanes(Matrix inverseViewProjection, DirectX::XMVECTOR* NearPlane, DirectX::XMVECTOR* FarPlane, DirectX::XMVECTOR* RightPlane, DirectX::XMVECTOR* LeftPlane, DirectX::XMVECTOR* TopPlane, DirectX::XMVECTOR* BottomPlane)
	{
		Vector4 corners[8] =
		{
			Vector4(-1.0f, 1.0f, 0.0f, 1.0f),
			Vector4(1.0f, 1.0f, 0.0f, 1.0f),
			Vector4(1.0f, -1.0f, 0.0f, 1.0f),
			Vector4(-1.0f, -1.0f, 0.0f, 1.0f),

			Vector4(-1.0f, 1.0f, 1.0f, 1.0f),
			Vector4(1.0f, 1.0f, 1.0f, 1.0f),
			Vector4(1.0f, -1.0f, 1.0f, 1.0f),
			Vector4(-1.0f, -1.0f, 1.0f, 1.0f),
		};

		for (int i = 0; i < 8; ++i)
		{
			corners[i] = Vector4::Transform(corners[i], inverseViewProjection);
			corners[i].x /= corners[i].w;
			corners[i].y /= corners[i].w;
			corners[i].z /= corners[i].w;
		}

		*NearPlane = DirectX::XMPlaneNormalize(DirectX::XMPlaneFromPoints(corners[0], corners[1], corners[2]));
		*FarPlane = DirectX::XMPlaneNormalize(DirectX::XMPlaneFromPoints(corners[6], corners[5], corners[4]));

		*LeftPlane = DirectX::XMPlaneNormalize(DirectX::XMPlaneFromPoints(corners[0], corners[3], corners[7]));
		*RightPlane = DirectX::XMPlaneNormalize(DirectX::XMPlaneFromPoints(corners[6], corners[2], corners[1]));

		*TopPlane = DirectX::XMPlaneNormalize(DirectX::XMPlaneFromPoints(corners[4], corners[1], corners[0]));
		*BottomPlane = DirectX::XMPlaneNormalize(DirectX::XMPlaneFromPoints(corners[7], corners[3], corners[2]));
	}

	inline uint32_t GetMipCount(const uint32_t& width, const uint32_t& height, const bool& generateMips)
	{
		if (generateMips)
		{
			uint32_t mipCount = static_cast<uint32_t>(log2(std::min(static_cast<float>(width), static_cast<float>(height))));
			// Based on https://stackoverflow.com/questions/108318/how-can-i-test-whether-a-number-is-a-power-of-2
			if ((width & (width - 1)) == 0 && (height & (height - 1)) == 0)
			{
				return mipCount;
			}
		}
		return 1;
	}

	inline float GetRandomFloat01()
	{
		return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}

	inline float GetRandomFloat(float a, float b)
	{
		return a + GetRandomFloat01()*(b - a);
	}
}