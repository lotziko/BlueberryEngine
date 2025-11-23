#pragma once

#include "..\VecMath.h"
#include "..\Matrix.h"

static __forceinline__ __device__ bool isValid(float3& vector)
{
	return isfinite(vector.x) && isfinite(vector.y) && isfinite(vector.z);
}

static __forceinline__ __device__ void cosineSampleHemisphere(const float u1, const float u2, float3& p)
{
	// Uniformly sample disk.
	const float r = sqrtf(u1);
	const float phi = 2.0f * M_PIf * u2;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);

	// Project up to hemisphere.
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

static __forceinline__ __device__ void cosineSampleSphere(const float u1, const float u2, float3& p)
{
	float z = 1.0f - 2.0f * u1;
	float phi = 2.0f * M_PIf * u2;
	float r = sqrtf(1.0f - z * z);

	p.x = r * cosf(phi);
	p.y = r * sinf(phi);
	p.z = z;
}

static __forceinline__ __device__ void transformOSToWS(Matrix3x4 localToWorld, float3 normalOS, float4 tangentOS, float3 &normalWS, float3 &tangentWS, float3 &bitangentWS)
{
	normalWS = normalize(localToWorld * make_float4(normalOS.x, normalOS.y, normalOS.z, 0));
	tangentWS = normalize(localToWorld * make_float4(tangentOS.x, tangentOS.y, tangentOS.z, 0));
	bitangentWS = cross(normalWS, tangentWS) * tangentOS.w;
}

// Based on DirectXTex XMFLOAT3PK
inline __forceinline__ __device__ unsigned int convertFloatToR11G11B10(float r, float g, float b)
{
	unsigned int iValue[3] = { __float_as_uint(r), __float_as_uint(g), __float_as_uint(b) };
	unsigned int result[3];

	// X & Y Channels (5-bit exponent, 6-bit mantissa)
	for (unsigned int j = 0; j < 2; ++j)
	{
		unsigned int sign = iValue[j] & 0x80000000;
		unsigned int i = iValue[j] & 0x7FFFFFFF;

		if ((i & 0x7F800000) == 0x7F800000)
		{
			// INF or NAN
			result[j] = 0x7C0U;
			if ((i & 0x7FFFFF) != 0)
			{
				result[j] = 0x7FFU;
			}
			else if (sign)
			{
				// -INF is clamped to 0 since 3PK is positive only
				result[j] = 0;
			}
		}
		else if (sign || i < 0x35800000)
		{
			// 3PK is positive only, so clamp to zero
			result[j] = 0;
		}
		else if (i > 0x477E0000U)
		{
			// The number is too large to be represented as a float11, set to max
			result[j] = 0x7BFU;
		}
		else
		{
			if (i < 0x38800000U)
			{
				// The number is too small to be represented as a normalized float11
				// Convert it to a denormalized value.
				unsigned int Shift = 113U - (i >> 23U);
				i = (0x800000U | (i & 0x7FFFFFU)) >> Shift;
			}
			else
			{
				// Rebias the exponent to represent the value as a normalized float11
				i += 0xC8000000U;
			}

			result[j] = ((i + 0xFFFFU + ((i >> 17U) & 1U)) >> 17U) & 0x7ffU;
		}
	}

	// Z Channel (5-bit exponent, 5-bit mantissa)
	unsigned int sign = iValue[2] & 0x80000000;
	unsigned int i = iValue[2] & 0x7FFFFFFF;

	if ((i & 0x7F800000) == 0x7F800000)
	{
		// INF or NAN
		result[2] = 0x3E0U;
		if (i & 0x7FFFFF)
		{
			result[2] = 0x3FFU;
		}
		else if (sign || i < 0x36000000)
		{
			// -INF is clamped to 0 since 3PK is positive only
			result[2] = 0;
		}
	}
	else if (sign)
	{
		// 3PK is positive only, so clamp to zero
		result[2] = 0;
	}
	else if (i > 0x477C0000U)
	{
		// The number is too large to be represented as a float10, set to max
		result[2] = 0x3DFU;
	}
	else
	{
		if (i < 0x38800000U)
		{
			// The number is too small to be represented as a normalized float10
			// Convert it to a denormalized value.
			unsigned int shift = 113U - (i >> 23U);
			i = (0x800000U | (i & 0x7FFFFFU)) >> shift;
		}
		else
		{
			// Rebias the exponent to represent the value as a normalized float10
			i += 0xC8000000U;
		}

		result[2] = ((i + 0x1FFFFU + ((i >> 18U) & 1U)) >> 18U) & 0x3ffU;
	}

	// Pack result into memory
	return (result[0] & 0x7ff) | ((result[1] & 0x7ff) << 11) | ((result[2] & 0x3ff) << 22);
}