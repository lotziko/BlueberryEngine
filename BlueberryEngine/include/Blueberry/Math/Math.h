#pragma once

#include "SimpleMath.h"

#include <algorithm>

namespace Blueberry
{
#undef max
#undef min
	struct Vector2Int : public DirectX::XMINT2
	{
		Vector2Int() = default;

		Vector2Int(const Vector2Int&) = default;
		Vector2Int& operator=(const Vector2Int&) = default;

		Vector2Int(Vector2Int&&) = default;
		Vector2Int& operator=(Vector2Int&&) = default;

		constexpr Vector2Int(int32_t _x, int32_t _y) noexcept : DirectX::XMINT2(_x, _y) {}
		explicit Vector2Int(_In_reads_(2) const int32_t * pArray) noexcept : DirectX::XMINT2(pArray) {}

		inline bool operator == (const Vector2Int& V) const noexcept
		{
			using namespace DirectX;
			XMVECTOR v1 = XMLoadSInt2(this);
			XMVECTOR v2 = XMLoadSInt2(&V);
			return XMVector2Equal(v1, v2);
		}

		inline bool operator != (const Vector2Int& V) const noexcept
		{
			using namespace DirectX;
			XMVECTOR v1 = XMLoadSInt2(this);
			XMVECTOR v2 = XMLoadSInt2(&V);
			return XMVector2NotEqual(v1, v2);
		}

		inline Vector2Int& operator+= (const Vector2Int& V) noexcept
		{
			using namespace DirectX;
			XMVECTOR v1 = XMLoadSInt2(this);
			XMVECTOR v2 = XMLoadSInt2(&V);
			XMVECTOR X = XMVectorAdd(v1, v2);
			XMStoreSInt2(this, X);
			return *this;
		}
	};

	inline Vector2Int operator+ (const Vector2Int& V1, const Vector2Int& V2) noexcept
	{
		using namespace DirectX;
		XMVECTOR v1 = XMLoadSInt2(&V1);
		XMVECTOR v2 = XMLoadSInt2(&V2);
		XMVECTOR X = XMVectorAdd(v1, v2);
		Vector2Int R;
		XMStoreSInt2(&R, X);
		return R;
	}

	struct Vector3Int : public DirectX::XMINT3
	{
		Vector3Int() = default;

		Vector3Int(const Vector3Int&) = default;
		Vector3Int& operator=(const Vector3Int&) = default;

		Vector3Int(Vector3Int&&) = default;
		Vector3Int& operator=(Vector3Int&&) = default;

		constexpr Vector3Int(int32_t _x, int32_t _y, int32_t _z) noexcept : DirectX::XMINT3(_x, _y, _z) {}
		explicit Vector3Int(_In_reads_(3) const int32_t * pArray) noexcept : DirectX::XMINT3(pArray) {}

		inline bool operator == (const Vector3Int& V) const noexcept
		{
			using namespace DirectX;
			XMVECTOR v1 = XMLoadSInt3(this);
			XMVECTOR v2 = XMLoadSInt3(&V);
			return XMVector2Equal(v1, v2);
		}

		inline bool operator != (const Vector3Int& V) const noexcept
		{
			using namespace DirectX;
			XMVECTOR v1 = XMLoadSInt3(this);
			XMVECTOR v2 = XMLoadSInt3(&V);
			return XMVector3NotEqual(v1, v2);
		}

		inline Vector3Int& operator+= (const Vector3Int& V) noexcept
		{
			using namespace DirectX;
			XMVECTOR v1 = XMLoadSInt3(this);
			XMVECTOR v2 = XMLoadSInt3(&V);
			XMVECTOR X = XMVectorAdd(v1, v2);
			XMStoreSInt3(this, X);
			return *this;
		}
	};

	inline Vector3Int operator+ (const Vector3Int& V1, const Vector3Int& V2) noexcept
	{
		using namespace DirectX;
		XMVECTOR v1 = XMLoadSInt3(&V1);
		XMVECTOR v2 = XMLoadSInt3(&V2);
		XMVECTOR X = XMVectorAdd(v1, v2);
		Vector3Int R;
		XMStoreSInt3(&R, X);
		return R;
	}

	struct Vector4Int : public DirectX::XMINT4
	{
		Vector4Int() = default;

		Vector4Int(const Vector4Int&) = default;
		Vector4Int& operator=(const Vector4Int&) = default;

		Vector4Int(Vector4Int&&) = default;
		Vector4Int& operator=(Vector4Int&&) = default;

		constexpr Vector4Int(int32_t _x, int32_t _y, int32_t _z, int32_t _w) noexcept : DirectX::XMINT4(_x, _y, _z, _w) {}
		explicit Vector4Int(_In_reads_(4) const int32_t* pArray) noexcept : DirectX::XMINT4(pArray) {}

		inline bool operator == (const Vector4Int& V) const noexcept
		{
			using namespace DirectX;
			XMVECTOR v1 = XMLoadSInt4(this);
			XMVECTOR v2 = XMLoadSInt4(&V);
			return XMVector2Equal(v1, v2);
		}

		inline bool operator != (const Vector4Int& V) const noexcept
		{
			using namespace DirectX;
			XMVECTOR v1 = XMLoadSInt4(this);
			XMVECTOR v2 = XMLoadSInt4(&V);
			return XMVector4NotEqual(v1, v2);
		}

		inline Vector4Int& operator+= (const Vector4Int& V) noexcept
		{
			using namespace DirectX;
			XMVECTOR v1 = XMLoadSInt4(this);
			XMVECTOR v2 = XMLoadSInt4(&V);
			XMVECTOR X = XMVectorAdd(v1, v2);
			XMStoreSInt4(this, X);
			return *this;
		}
	};

	inline Vector4Int operator+ (const Vector4Int& V1, const Vector4Int& V2) noexcept
	{
		using namespace DirectX;
		XMVECTOR v1 = XMLoadSInt4(&V1);
		XMVECTOR v2 = XMLoadSInt4(&V2);
		XMVECTOR X = XMVectorAdd(v1, v2);
		Vector4Int R;
		XMStoreSInt4(&R, X);
		return R;
	}

	struct Color : public DirectX::SimpleMath::Color
	{
		Color() noexcept : DirectX::SimpleMath::Color(0.f, 0.f, 0.f, 1.f) {}

		Color(float red, float green, float blue, float alpha = 1.f) noexcept : DirectX::SimpleMath::Color(red, green, blue, alpha) {}

		inline Color ToLinear() const { return Color(std::powf(x, 2.2f), std::powf(y, 2.2f), std::powf(z, 2.2f), w); }
	};

	struct AABB : public DirectX::BoundingBox
	{
		AABB() noexcept = default;

		AABB(const AABB&) = default;
		AABB& operator=(const AABB&) = default;

		AABB(AABB&&) = default;
		AABB& operator=(AABB&&) = default;

		AABB(_In_ const DirectX::XMFLOAT3& center, _In_ const DirectX::XMFLOAT3& extents) noexcept
			: DirectX::BoundingBox(center, extents) {}

		bool operator == (const AABB& aabb) const noexcept
		{
			using XMVECTOR = DirectX::XMVECTOR;
			XMVECTOR c1 = DirectX::XMLoadFloat3(&Center);
			XMVECTOR e1 = DirectX::XMLoadFloat3(&Extents);

			XMVECTOR c2 = DirectX::XMLoadFloat3(&aabb.Center);
			XMVECTOR e2 = DirectX::XMLoadFloat3(&aabb.Extents);

			return (DirectX::XMVector3Equal(c1, c2) && DirectX::XMVector3Equal(e1, e2)) != 0;
		}

		bool operator != (const AABB& aabb) const noexcept
		{
			return !(*this == aabb);
		}
	};

	using Vector2 = DirectX::SimpleMath::Vector2;
	using Vector3 = DirectX::SimpleMath::Vector3;
	using Vector4 = DirectX::SimpleMath::Vector4;
	using Vector2Uint = DirectX::XMUINT2;
	using Vector3Uint = DirectX::XMUINT3;
	using Vector4Uint = DirectX::XMUINT4;
	using Quaternion = DirectX::SimpleMath::Quaternion;
	using Ray = DirectX::SimpleMath::Ray;
	using Matrix = DirectX::SimpleMath::Matrix;
	using Rectangle = DirectX::SimpleMath::Rectangle;
	using Viewport = DirectX::SimpleMath::Viewport;
	using OBB = DirectX::BoundingOrientedBox;
	using Sphere = DirectX::BoundingSphere;
	using Frustum = DirectX::BoundingFrustum;

	struct TRS
	{
		Vector3 position;
		Quaternion rotation;
		Vector3 scale;
	};

	struct RectangleFloat
	{
		float x;
		float y;
		float width;
		float height;
	};

	class Math
	{
	public:
		static inline float Sign(float f)
		{
			return f >= 0.0f ? 1.0f : -1.0f;
		}

		static inline float ToDegrees(float radians)
		{
			return radians * DegreeToRad;
		}

		static inline Vector3 ToDegrees(Vector3 radians)
		{
			return Vector3(radians.x * DegreeToRad, radians.y * DegreeToRad, radians.z * DegreeToRad);
		}

		static inline float ToRadians(float degrees)
		{
			return degrees * RadToDegree;
		}

		static inline Vector3 ToRadians(Vector3 degrees)
		{
			return Vector3(degrees.x * RadToDegree, degrees.y * RadToDegree, degrees.z * RadToDegree);
		}

		static inline float Lerp(float a, float b, float t)
		{
			return a + (b - a) * t;
		}

		static inline float MoveTowards(float from, float to, float maxLength)
		{
			if (std::abs(to - from) <= maxLength)
			{
				return to;
			}
			return from + Sign(to - from) * maxLength;
		}

		static inline Vector3 MoveTowards(Vector3 from, Vector3 to, float maxLength)
		{
			Vector3 delta = to - from;
			float sqrLength = delta.LengthSquared();
			if (sqrLength == 0 || (maxLength >= 0 && sqrLength <= maxLength * maxLength))
			{
				return to;
			}
			float magnitude = (float)std::sqrtf(sqrLength);
			return from + delta / magnitude * maxLength;
		}

		static inline Vector2 ClampMagnitude(Vector2 value, float maxLength)
		{
			float sqrLength = value.LengthSquared();
			if (sqrLength > maxLength * maxLength)
			{
				float magnitude = (float)std::sqrtf(sqrLength);
				float normalizedX = value.x / magnitude;
				float normalizedY = value.y / magnitude;
				return Vector2(normalizedX * maxLength, normalizedY * maxLength);
			}
			return value;
		}

		static inline Vector3 Round(Vector3 vector)
		{
			return Vector3(std::roundf(vector.x), std::roundf(vector.y), std::roundf(vector.z));
		}

		static inline Vector3 RoundToN(Vector3 value, uint32_t n)
		{
			float powerOf10 = std::pow(10.0f, static_cast<float>(n));
			value.x = std::roundf(value.x * powerOf10) / powerOf10;
			value.y = std::roundf(value.y * powerOf10) / powerOf10;
			value.z = std::roundf(value.z * powerOf10) / powerOf10;
			return value;
		}

		static inline Quaternion RoundToN(Quaternion value, uint32_t n)
		{
			float powerOf10 = std::pow(10.0f, static_cast<float>(n));
			value.x = std::roundf(value.x * powerOf10) / powerOf10;
			value.y = std::roundf(value.y * powerOf10) / powerOf10;
			value.z = std::roundf(value.z * powerOf10) / powerOf10;
			value.w = std::roundf(value.w * powerOf10) / powerOf10;
			return value;
		}

		// https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
		static inline uint32_t NextPowerOfTwo(uint32_t value)
		{
			value--;
			value |= value >> 1;
			value |= value >> 2;
			value |= value >> 4;
			value |= value >> 8;
			value |= value >> 16;
			value++;
			return value;
		}

		static inline int32_t NextDivisableBy(int32_t value, int32_t by)
		{
			return (value + by - 1) & ~(by - 1);
		}

		static inline uint32_t NextDivisableBy(uint32_t value, uint32_t by)
		{
			return (value + by - 1) & ~(by - 1);
		}

		static inline int64_t NextDivisableBy(int64_t value, int64_t by)
		{
			return (value + by - 1) & ~(by - 1);
		}

		static inline uint64_t NextDivisableBy(uint64_t value, uint64_t by)
		{
			return (value + by - 1) & ~(by - 1);
		}

		static inline bool Approximately(float a, float b)
		{
			return std::abs(b - a) < std::max(0.000001f * std::max(std::abs(a), std::abs(b)), Epsilon * 8.0f);
		}

		// TODO intrinsics
		static inline bool Approximately(Vector3 a, Vector3 b)
		{
			return Approximately(a.x, b.x) && Approximately(a.y, b.y) && Approximately(a.z, b.z);
		}

		static inline bool Approximately(Quaternion a, Quaternion b)
		{
			return Approximately(a.x, b.x) && Approximately(a.y, b.y) && Approximately(a.z, b.z) && Approximately(a.w, b.w);
		}

		static inline Matrix CreateTRS(Vector3 position, Quaternion rotation, Vector3 scale)
		{
			return Matrix::CreateScale(scale) * Matrix::CreateFromQuaternion(rotation) * Matrix::CreateTranslation(position);
		}

		static inline Quaternion LookRotation(Vector3 forward, Vector3 up = Vector3::UnitY)
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

		static inline Vector3 MultiplyPoint(Matrix matrix, Vector3 point)
		{
			Vector4 pointToTransform = Vector4(point.x, point.y, point.z, 1.0f);
			pointToTransform = Vector4::Transform(pointToTransform, matrix);
			pointToTransform.w = 1 / pointToTransform.w;
			pointToTransform.x *= pointToTransform.w;
			pointToTransform.y *= pointToTransform.w;
			pointToTransform.z *= pointToTransform.w;
			return Vector3(pointToTransform.x, pointToTransform.y, pointToTransform.z);
		}

		static inline Vector3 MultiplyVector(Matrix matrix, Vector3 vector)
		{
			Vector4 vectorToTransform = Vector4(vector.x, vector.y, vector.z, 0.0f);
			vectorToTransform = Vector4::Transform(vectorToTransform, matrix);
			return Vector3(vectorToTransform.x, vectorToTransform.y, vectorToTransform.z);
		}

		static inline DirectX::ContainmentType ContainsFlipped(const DirectX::BoundingFrustum& frustum, const DirectX::BoundingBox& box)
		{
			using XMVECTOR = DirectX::XMVECTOR;
			// Load origin and orientation of the frustum.
			XMVECTOR vOrigin = DirectX::XMLoadFloat3(&frustum.Origin);
			XMVECTOR vOrientation = DirectX::XMLoadFloat4(&frustum.Orientation);

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

		static inline void GetOrthographicPlanes(Matrix inverseViewProjection, DirectX::XMVECTOR* NearPlane, DirectX::XMVECTOR* FarPlane, DirectX::XMVECTOR* RightPlane, DirectX::XMVECTOR* LeftPlane, DirectX::XMVECTOR* TopPlane, DirectX::XMVECTOR* BottomPlane)
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

		static inline uint32_t GetMipCount(const uint32_t& width, const uint32_t& height, const bool& generateMips)
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

		static inline float GetRandomFloat01()
		{
			return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		}

		static inline float GetRandomFloat(float a, float b)
		{
			return a + GetRandomFloat01()*(b - a);
		}

		static inline int GetRandomInt(int a, int b)
		{
			return a + rand() % (b - a);
		}

		// https://github.com/Auburn/FastNoiseLite/blob/master/Cpp/FastNoiseLite.h
		static inline float PerlinNoise(float x, float y)
		{
			int seed = 0;
			int x0 = FastFloor(x);
			int y0 = FastFloor(y);

			float xd0 = (float)(x - x0);
			float yd0 = (float)(y - y0);
			float xd1 = xd0 - 1;
			float yd1 = yd0 - 1;

			float xs = InterpQuintic(xd0);
			float ys = InterpQuintic(yd0);

			x0 *= PrimeX;
			y0 *= PrimeY;
			int x1 = x0 + PrimeX;
			int y1 = y0 + PrimeY;

			float xf0 = Lerp(GradCoord(seed, x0, y0, xd0, yd0), GradCoord(seed, x1, y0, xd1, yd0), xs);
			float xf1 = Lerp(GradCoord(seed, x0, y1, xd0, yd1), GradCoord(seed, x1, y1, xd1, yd1), xs);

			return (Lerp(xf0, xf1, ys) * 1.4247691104677813f) * 0.5f + 0.5f;
		}

		static inline float Pi = 3.1415926535f;
		static inline float DegreeToRad = 57.29577951471995f;
		static inline float RadToDegree = 0.0174532925194444f;
		static inline float Epsilon = 1.4e-45f;

	private:
		static inline int FastFloor(float f)
		{
			return f >= 0 ? (int)f : (int)f - 1;
		}

		static inline float InterpQuintic(float t)
		{
			return t * t * t * (t * (t * 6 - 15) + 10);
		}

		static inline int Hash(int seed, int xPrimed, int yPrimed)
		{
			int hash = seed ^ xPrimed ^ yPrimed;

			hash *= 0x27d4eb2d;
			return hash;
		}

		static inline float GradCoord(int seed, int xPrimed, int yPrimed, float xd, float yd)
		{
			int hash = Hash(seed, xPrimed, yPrimed);
			hash ^= hash >> 15;
			hash &= 127 << 1;

			float xg = s_Gradients2D[hash];
			float yg = s_Gradients2D[hash | 1];

			return xd * xg + yd * yg;
		}

	private:
		static inline int PrimeX = 501125321;
		static inline int PrimeY = 1136930381;
		static inline int PrimeZ = 1720413743;

		static inline float s_Gradients2D[] =
		{
			0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f, 0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f,
			0.923879532511287f, 0.38268343236509f, 0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f, 0.130526192220052f, -0.99144486137381f,
			-0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
			-0.923879532511287f, -0.38268343236509f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f, -0.130526192220052f, 0.99144486137381f,
			0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f, 0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f,
			0.923879532511287f, 0.38268343236509f, 0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f, 0.130526192220052f, -0.99144486137381f,
			-0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
			-0.923879532511287f, -0.38268343236509f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f, -0.130526192220052f, 0.99144486137381f,
			0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f, 0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f,
			0.923879532511287f, 0.38268343236509f, 0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f, 0.130526192220052f, -0.99144486137381f,
			-0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
			-0.923879532511287f, -0.38268343236509f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f, -0.130526192220052f, 0.99144486137381f,
			0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f, 0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f,
			0.923879532511287f, 0.38268343236509f, 0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f, 0.130526192220052f, -0.99144486137381f,
			-0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
			-0.923879532511287f, -0.38268343236509f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f, -0.130526192220052f, 0.99144486137381f,
			0.130526192220052f, 0.99144486137381f, 0.38268343236509f, 0.923879532511287f, 0.608761429008721f, 0.793353340291235f, 0.793353340291235f, 0.608761429008721f,
			0.923879532511287f, 0.38268343236509f, 0.99144486137381f, 0.130526192220051f, 0.99144486137381f, -0.130526192220051f, 0.923879532511287f, -0.38268343236509f,
			0.793353340291235f, -0.60876142900872f, 0.608761429008721f, -0.793353340291235f, 0.38268343236509f, -0.923879532511287f, 0.130526192220052f, -0.99144486137381f,
			-0.130526192220052f, -0.99144486137381f, -0.38268343236509f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
			-0.923879532511287f, -0.38268343236509f, -0.99144486137381f, -0.130526192220052f, -0.99144486137381f, 0.130526192220051f, -0.923879532511287f, 0.38268343236509f,
			-0.793353340291235f, 0.608761429008721f, -0.608761429008721f, 0.793353340291235f, -0.38268343236509f, 0.923879532511287f, -0.130526192220052f, 0.99144486137381f,
			0.38268343236509f, 0.923879532511287f, 0.923879532511287f, 0.38268343236509f, 0.923879532511287f, -0.38268343236509f, 0.38268343236509f, -0.923879532511287f,
			-0.38268343236509f, -0.923879532511287f, -0.923879532511287f, -0.38268343236509f, -0.923879532511287f, 0.38268343236509f, -0.38268343236509f, 0.923879532511287f,
		};
	};
}

template <>
struct std::hash<Blueberry::Vector2Int>
{
	size_t operator()(const Blueberry::Vector2Int& vector) const
	{
		return (static_cast<size_t>(vector.x) << 32) ^ static_cast<size_t>(vector.y);
	}
};

template <>
struct std::hash<Blueberry::Vector3Int>
{
	size_t operator()(const Blueberry::Vector3Int& vector) const
	{
		return (static_cast<size_t>(vector.x) << 32) ^ (static_cast<size_t>(vector.y) << 16) ^ static_cast<size_t>(vector.z);
	}
};

template <>
struct std::hash<Blueberry::Vector3>
{
	size_t operator()(const Blueberry::Vector3& vector) const
	{
		return (static_cast<size_t>(vector.x) << 32) ^ (static_cast<size_t>(vector.y) << 16) ^ static_cast<size_t>(vector.z);
	}
};
