#include <optix.h>
#include "Params.h"

#include "VecMath.h"
#include "Random.h"
#include "Matrix.h"

namespace Blueberry
{
	extern "C" 
	{
		__constant__ Params params;
	}

	static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction, float2& jitter)
	{
		const float3 U = params.camU;
		const float3 V = params.camV;
		const float3 W = params.camW;
		const float2 d = 2.0f * make_float2(
			(static_cast<float>(idx.x) + jitter.x) / static_cast<float>(dim.x),
			(static_cast<float>(idx.y) + jitter.y) / static_cast<float>(dim.y)
		) - 1.0f;

		origin = params.camEye;
		direction = normalize(d.x * U + d.y * V + W);
	}

	static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
	{
		// Uniformly sample disk.
		const float r = sqrtf(u1);
		const float phi = 2.0f*M_PIf * u2;
		p.x = r * cosf(phi);
		p.y = r * sinf(phi);

		// Project up to hemisphere.
		p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x*p.x - p.y*p.y));
	}

	extern "C" __global__ void __raygen__default()
	{
		const uint3 idx = optixGetLaunchIndex();
		const uint3 dim = optixGetLaunchDimensions();

		const unsigned int index = idx.y * params.imageWidth + idx.x;
		unsigned int seed = tea<4>(index, params.accumulationFrameIndex);
		float2 subpixelJitter = make_float2(rnd(seed), rnd(seed));

		float3 rayOrigin, rayDirection;
		computeRay(idx, dim, rayOrigin, rayDirection, subpixelJitter);

		unsigned int depth = 0;
		unsigned int c0 = 0;
		unsigned int c1 = 0;
		unsigned int c2 = 0;
		optixTrace(
			params.handle,
			rayOrigin,
			rayDirection,
			0,											// Min intersection distance
			1e16f,										// Max intersection distance
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			0,											// SBT offset
			2,											// SBT stride 
			0,											// missSBTIndex
			depth, c0, c1, c2, seed);
		float3 color = make_float3(__uint_as_float(c0), __uint_as_float(c1), __uint_as_float(c2));
		params.accumulatedImage[index] += make_float4(color.x, color.y, color.z, 0);
		float4 resultColor = params.accumulatedImage[index] / (float)(params.accumulationFrameIndex + 1);
		params.image[index] = make_float4(resultColor.x, resultColor.y, resultColor.z, 1.0f);//make_color(make_float3(resultColor.x, resultColor.y, resultColor.z));
	}

	extern "C" __global__ void __miss__default()
	{
		optixSetPayload_0(0);
	}

	extern "C" __global__ void __closesthit__default()
	{
		HitGroupData* hitGroupData = (HitGroupData*)optixGetSbtDataPointer();

		const int primitiveIndex = optixGetPrimitiveIndex();
		const uint3 index = hitGroupData->indices[primitiveIndex];
		const float2 uv = optixGetTriangleBarycentrics();
		const float3 hitPos = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

		// Maybe do some check for normal and tangent not being null
		// Need to transform them to world space, now they are in object space, Sponza does not work
		float3 n0 = hitGroupData->normals[index.x];
		float3 n1 = hitGroupData->normals[index.y];
		float3 n2 = hitGroupData->normals[index.z];

		float4 t0 = hitGroupData->tangents[index.x];
		float4 t1 = hitGroupData->tangents[index.y];
		float4 t2 = hitGroupData->tangents[index.z];

		Matrix3x4 localToWorld = params.instanceMatrices[optixGetInstanceId()];
		float3 normal = normalize(n0 * (1 - uv.x - uv.y) + n1 * uv.x + n2 * uv.y);
		float4 tangent = normalize(make_float4(t0.x, t0.y, t0.z, t0.w) * (1 - uv.x - uv.y) + make_float4(t1.x, t1.y, t1.z, t1.w) * uv.x + make_float4(t2.x, t2.y, t2.z, t2.w) * uv.y);
		//float3 bitangent = cross(normal, tangent);

		float3 normalWS = localToWorld * make_float4(normal.x, normal.y, normal.z, 0);
		float3 tangentWS = localToWorld * make_float4(tangent.x, tangent.y, tangent.z, 0);
		float3 bitangentWS = cross(normalWS, tangentWS) * tangent.w;

		float3 rayOrigin = hitPos + normalWS * 0.001f;
		float3 radiance = make_float3(__uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()), __uint_as_float(optixGetPayload_3()));

		/*optixSetPayload_1(__float_as_uint(normalWS.x));
		optixSetPayload_2(__float_as_uint(normalWS.y));
		optixSetPayload_3(__float_as_uint(normalWS.z));
		return;*/

		// Next ray
		unsigned int depth = optixGetPayload_0();
		unsigned int seed = optixGetPayload_4();
		if (depth < BOUNCE_COUNT)
		{
			unsigned int newDepth = depth + 1;
			const float z1 = rnd(seed);
			const float z2 = rnd(seed);

			float3 randomDirection;
			cosine_sample_hemisphere(z1, z2, randomDirection);
			float3 rayDirection = randomDirection.x * tangentWS + randomDirection.y * bitangentWS + randomDirection.z * normalWS;
			unsigned int c0 = 0;
			unsigned int c1 = 0;
			unsigned int c2 = 0;
			optixTrace(
				params.handle,
				rayOrigin,
				rayDirection,
				0,											// Min intersection distance
				1e16f,										// Max intersection distance
				0.0f,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_NONE,
				0,											// SBT offset
				2,											// SBT stride 
				0,											// missSBTIndex
				newDepth, c0, c1, c2, seed);
			radiance += make_float3(__uint_as_float(c0), __uint_as_float(c1), __uint_as_float(c2));
		}

		if (depth > 0)
		{
			// Shadow ray
			const float3 shadowRayDirection = params.directionalLight.direction;
			unsigned int isShadow = 0;
			optixTrace(
				params.handle,
				rayOrigin,       // offset to avoid self-intersection
				shadowRayDirection,
				0.001f,
				1e16f,
				0.0f,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				1,								// SBT offset shadow ray
				2,								// SBT stride shadow ray
				0,								// miss shadow ray
				isShadow
			);
			if (!isShadow)
			{
				float NdotL = dot(normalWS, params.directionalLight.direction);
				float attenuation = clamp(NdotL, 0.0f, 1.0f);
				radiance += params.directionalLight.color * (attenuation);
			}
		}
		optixSetPayload_1(__float_as_uint(radiance.x));
		optixSetPayload_2(__float_as_uint(radiance.y));
		optixSetPayload_3(__float_as_uint(radiance.z));
	}

	extern "C" __global__ void __closesthit__shadow()
	{
		optixSetPayload_0(1);
	}
}