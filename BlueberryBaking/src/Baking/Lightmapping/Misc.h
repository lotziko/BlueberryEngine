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
	const float phi = 2.0f*M_PIf * u2;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);

	// Project up to hemisphere.
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x*p.x - p.y*p.y));
}

static __forceinline__ __device__ void transformOSToWS(Matrix3x4 localToWorld, float3 normalOS, float4 tangentOS, float3 &normalWS, float3 &tangentWS, float3 &bitangentWS)
{
	normalWS = normalize(localToWorld * make_float4(normalOS.x, normalOS.y, normalOS.z, 0));
	tangentWS = normalize(localToWorld * make_float4(tangentOS.x, tangentOS.y, tangentOS.z, 0));
	bitangentWS = cross(normalWS, tangentWS) * tangentOS.w;
}
