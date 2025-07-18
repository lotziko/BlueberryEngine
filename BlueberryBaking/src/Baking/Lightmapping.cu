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
		normalWS = localToWorld * make_float4(normalOS.x, normalOS.y, normalOS.z, 0);
		tangentWS = localToWorld * make_float4(tangentOS.x, tangentOS.y, tangentOS.z, 0);
		bitangentWS = cross(normalWS, tangentWS) * tangentOS.w;
	}

	static __forceinline__ __device__ void traceRadiance(float3 &radiance, float3 positionWS, float3 normalWS, float3 tangentWS, float3 bitangentWS, unsigned int depth, unsigned int seed)
	{
		// Next ray
		if (depth < BOUNCE_COUNT)
		{
			unsigned int newDepth = depth + 1;
			const float z1 = rnd(seed);
			const float z2 = rnd(seed);

			float3 randomDirection;
			cosineSampleHemisphere(z1, z2, randomDirection);
			float3 rayOrigin = positionWS;
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

		// Shadow ray
		if (depth > 0)
		{
			float3 rayOrigin = positionWS;
			float3 rayDirection = params.directionalLight.direction;
			unsigned int isShadow = 0;
			optixTrace(
				params.handle,
				rayOrigin,       // offset to avoid self-intersection
				rayDirection,
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
	}

	extern "C" __global__ void __raygen__default()
	{
		const uint3 idx = optixGetLaunchIndex();
		const uint3 dim = optixGetLaunchDimensions();

		float2 texelSize = make_float2(1.0f / params.imageWidth, 1.0f / params.imageHeight);
		float4 result = {};

		const int sampleCount = 1024;
		int validSamples = 0;
		unsigned int imageIndex = idx.y * params.imageWidth + idx.x;
		unsigned int seed = tea<4>(imageIndex, params.accumulationFrameIndex);

		for (int i = 0; i < sampleCount; ++i)
		{
			float2 jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
			float2 samplePosition = make_float2((idx.x + 0.5f + jitter.x) * texelSize.x, (idx.y + 0.5f + jitter.y) * texelSize.y);
			BVHTriangle triangle = params.bvh.GetTriangle(samplePosition);
			if (triangle.index.y != UINT_MAX)
			{
				BVHInstance instance = params.bvh.instances[triangle.index.x];
				Matrix3x4 localToWorld = params.instanceMatrices[triangle.index.x];

				uint3 index = instance.indices[triangle.index.y / 3];
				float3 uvw = GetBarycentrics(samplePosition, triangle.p1, triangle.p2, triangle.p3);

				float3 v0 = instance.vertices[index.x];
				float3 v1 = instance.vertices[index.y];
				float3 v2 = instance.vertices[index.z];

				float3 n0 = instance.normals[index.x];
				float3 n1 = instance.normals[index.y];
				float3 n2 = instance.normals[index.z];

				float4 t0 = instance.tangents[index.x];
				float4 t1 = instance.tangents[index.y];
				float4 t2 = instance.tangents[index.z];

				float3 positionOS = v0 * uvw.z + v1 * uvw.x + v2 * uvw.y;
				float3 normalOS = n0 * uvw.z + n1 * uvw.x + n2 * uvw.y;
				float4 tangentOS = t0 * uvw.z + t1 * uvw.x + t2 * uvw.y;

				float3 positionWS = localToWorld * make_float4(positionOS.x, positionOS.y, positionOS.z, 1);
				float3 normalWS, tangentWS, bitangentWS;
				transformOSToWS(localToWorld, normalOS, tangentOS, normalWS, tangentWS, bitangentWS);

				float3 radiance = {};
				traceRadiance(radiance, positionWS + normalWS * 0.001f, normalWS, tangentWS, bitangentWS, 0, seed);
				result += make_float4(radiance.x, radiance.y, radiance.z, 1);
				validSamples += 1;
			}
		}
		params.image[imageIndex] = result / max(validSamples, 1);
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
		const float3 uvw = make_float3(uv.x, uv.y, (1 - uv.x - uv.y));
		const float3 hitPos = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

		// Maybe do some check for normal and tangent not being null
		float3 n0 = hitGroupData->normals[index.x];
		float3 n1 = hitGroupData->normals[index.y];
		float3 n2 = hitGroupData->normals[index.z];

		float4 t0 = hitGroupData->tangents[index.x];
		float4 t1 = hitGroupData->tangents[index.y];
		float4 t2 = hitGroupData->tangents[index.z];

		float3 normalOS = n0 * uvw.z + n1 * uvw.x + n2 * uvw.y;
		float4 tangentOS = make_float4(t0.x, t0.y, t0.z, t0.w) * uvw.z + make_float4(t1.x, t1.y, t1.z, t1.w) * uvw.x + make_float4(t2.x, t2.y, t2.z, t2.w) * uvw.y;
		
		float3 normalWS, tangentWS, bitangentWS;
		transformOSToWS(params.instanceMatrices[optixGetInstanceId()], normalOS, tangentOS, normalWS, tangentWS, bitangentWS);

		float3 radiance = make_float3(__uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()), __uint_as_float(optixGetPayload_3()));
		traceRadiance(radiance, hitPos + normalWS * 0.001f, normalWS, tangentWS, bitangentWS, optixGetPayload_0(), optixGetPayload_4());
		optixSetPayload_1(__float_as_uint(radiance.x));
		optixSetPayload_2(__float_as_uint(radiance.y));
		optixSetPayload_3(__float_as_uint(radiance.z));
	}

	extern "C" __global__ void __closesthit__shadow()
	{
		optixSetPayload_0(1);
	}
}