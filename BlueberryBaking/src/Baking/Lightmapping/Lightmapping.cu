#include <optix.h>
#include "LightmappingParams.h"

#include "..\VecMath.h"
#include "..\Random.h"
#include "..\Matrix.h"

namespace Blueberry
{
	extern "C" 
	{
		__constant__ LightmappingParams params;
	}

	static __forceinline__ __device__ bool isValid(float3& vector)
	{
		return isfinite(vector.x) && isfinite(vector.y) && isfinite(vector.z);
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
		normalWS = normalize(localToWorld * make_float4(normalOS.x, normalOS.y, normalOS.z, 0));
		tangentWS = normalize(localToWorld * make_float4(tangentOS.x, tangentOS.y, tangentOS.z, 0));
		bitangentWS = cross(normalWS, tangentWS) * tangentOS.w;
	}

	// Based on https://ndotl.wordpress.com/2018/08/29/baking-artifact-free-lightmaps/
	static __forceinline__ __device__ bool traceBackface(float3& positionWS, float3 normalWS, float3 tangentWS, float3 bitangentWS)
	{
		float maxDistance = (1.0f / params.texelPerUnit) * 0.5f;
		float3 rayOrigin = positionWS + normalWS * 0.001f;
		float3 rayDirections[4] = { tangentWS, -tangentWS, bitangentWS, -bitangentWS };
		for (int i = 0; i < 4; ++i)
		{
			unsigned int valid = 0;
			unsigned int c0 = 0;
			unsigned int c1 = 0;
			unsigned int c2 = 0;
			if (!isValid(rayOrigin))
			{
				continue;
			}
			optixTrace(
				params.handle,
				rayOrigin,
				rayDirections[i],
				0,											// Min intersection distance
				maxDistance,								// Max intersection distance
				0.0f,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_NONE,
				0,											// SBT offset
				3,											// SBT stride 
				0,											// missSBTIndex
				valid, c0, c1, c2);
			if (valid == 1)
			{
				float3 newPositionWS = make_float3(__uint_as_float(c0), __uint_as_float(c1), __uint_as_float(c2));
				if (isValid(newPositionWS))
				{
					positionWS = newPositionWS;
					return true;
				}
			}
		}
		return false;
	}

	static __forceinline__ __device__ bool traceRadiance(float3 &radiance, float3 positionWS, float3 normalWS, float3 tangentWS, float3 bitangentWS, unsigned int depth, unsigned int seed)
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
			if (!isValid(rayOrigin))
			{
				return false;
			}
			float3 rayDirection = randomDirection.x * tangentWS + randomDirection.y * bitangentWS + randomDirection.z * normalWS;
			unsigned int valid = 1;
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
				1,											// SBT offset
				3,											// SBT stride 
				1,											// missSBTIndex
				valid, newDepth, c0, c1, c2, seed);
			if (valid == 0)
			{
				return false;
			}
			radiance += make_float3(__uint_as_float(c0), __uint_as_float(c1), __uint_as_float(c2));
		}

		// Shadow ray
		if (depth > 0)
		{
			float3 rayOrigin = positionWS;
			if (!isValid(rayOrigin))
			{
				return false;
			}
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
				2,								// SBT offset shadow ray
				3,								// SBT stride shadow ray
				2,								// miss shadow ray
				isShadow
			);
			if (!isShadow)
			{
				float NdotL = dot(normalWS, params.directionalLight.direction);
				float attenuation = clamp(NdotL, 0.0f, 1.0f);
				radiance += params.directionalLight.color * (attenuation);
			}
		}
		return true;
	}

	struct  __align__(16) SamplePositionData
	{
		float3 positionWS;
		float3 normalWS;
		float3 tangentWS;
		float3 bitangentWS;
	};

	#define BATCH_SIZE 64
	#define RGB_TO_LUMINANCE make_float3(0.2125f, 0.7154f, 0.0721f)

	extern "C" __global__ void __raygen__radiance()
	{
		while (true)
		{
			unsigned int texelIndex = UINT_MAX;
			for (unsigned int i = 0; i < params.validTexelsCount; ++i)
			{
				if (atomicCAS(&params.complete[i], 0, 1) == 0)
				{
					texelIndex = i;
					break;
				}
			}

			if (texelIndex == UINT_MAX)
			{
				break;
			}

			const uint2 pos = params.validTexels[texelIndex];

			unsigned int imageIndex = pos.y * params.imageSize.x + pos.x;
			unsigned int seed = tea<4>(imageIndex, params.accumulationFrameIndex);

			float2 texelSize = make_float2(1.0f / params.imageSize.x, 1.0f / params.imageSize.y);

			SamplePositionData samplePositions[64];
			unsigned int validPositionCount = 0;
			for (int i = 0; i < 64; ++i)
			{
				float2 jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
				float2 samplePosition = make_float2((pos.x + 0.5f + jitter.x) * texelSize.x, (pos.y + 0.5f + jitter.y) * texelSize.y);
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
					//traceBackface(positionWS, normalWS, tangentWS, bitangentWS);

					SamplePositionData positionData = {};
					positionData.positionWS = positionWS;
					positionData.normalWS = normalWS;
					positionData.tangentWS = tangentWS;
					positionData.bitangentWS = bitangentWS;
					samplePositions[validPositionCount] = positionData;
					validPositionCount += 1;
				}
			}

			const int maxSampleCount = params.samplePerTexel;
			const int minSampleCount = maxSampleCount / 20;

			float3 imageResult = {};
			float3 normalResult = {};

			float mean = {};
			float meanDistSquared = {};
			int sampleCount = 0;
			int validSampleCount = 0;

			while (true)
			{
				SamplePositionData positionData = samplePositions[rnd_range(seed, validPositionCount)];
				float3 radiance = {};
				if (traceRadiance(radiance, positionData.positionWS + positionData.normalWS * 0.001f, positionData.normalWS, positionData.tangentWS, positionData.bitangentWS, 0, seed))
				{
					validSampleCount += 1;
				}
				sampleCount += 1;
				imageResult += radiance;
				if (validSampleCount < sampleCount / 10)
				{
					break;
				}

				float sample = dot(radiance, RGB_TO_LUMINANCE);
				float delta = sample - mean;
				mean += delta / sampleCount;
				meanDistSquared += delta * (sample - mean);

				if (validSampleCount > minSampleCount)
				{
					float variance = meanDistSquared / (sampleCount - 1);
					float threshold = 0.05f / 1.96f;
					float standardError = sqrtf(variance / sampleCount);

					if (standardError < mean * threshold)
					{
						break;
					}
				}

				if (sampleCount >= maxSampleCount)
				{
					break;
				}
			}

			if (validSampleCount > maxSampleCount / 100) // Skip texels with low valid sample count and interpolate from nearby ones
			{
				params.color[imageIndex] = make_float4(imageResult / max(validSampleCount, 1), 1);
				params.normal[imageIndex] = make_float4(normalResult / max(validSampleCount, 1), 1);
			}
			atomicAdd(params.completeCounter, 1);
		}
	}

	extern "C" __global__ void __miss__backface()
	{
	}

	extern "C" __global__ void __miss__radiance()
	{
		if (dot(optixGetWorldRayDirection(), make_float3(0, 1, 0)) > 0)
		{
			optixSetPayload_2(__float_as_uint(params.directionalLight.color.x * 0.5));
			optixSetPayload_3(__float_as_uint(params.directionalLight.color.y * 0.5));
			optixSetPayload_4(__float_as_uint(params.directionalLight.color.z * 0.5));
		}
		else
		{
			optixSetPayload_0(0);
		}
	}

	extern "C" __global__ void __miss__shadow()
	{
	}

	extern "C" __global__ void __closesthit__backface()
	{
		if (optixIsBackFaceHit())
		{
			const float3 hitPos = optixGetWorldRayOrigin() + (optixGetRayTmax() + 0.001f) * optixGetWorldRayDirection();
			optixSetPayload_0(1);
			optixSetPayload_1(__float_as_uint(hitPos.x));
			optixSetPayload_2(__float_as_uint(hitPos.y));
			optixSetPayload_3(__float_as_uint(hitPos.z));
		}
	}

	extern "C" __global__ void __closesthit__radiance()
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

		// Avoid backface
		if (dot(normalWS, optixGetWorldRayDirection()) > 0.0f)
		{
			optixSetPayload_0(0);
			return;
		}

		float3 radiance = make_float3(__uint_as_float(optixGetPayload_2()), __uint_as_float(optixGetPayload_3()), __uint_as_float(optixGetPayload_4()));
		if (traceRadiance(radiance, hitPos + normalWS * 0.001f, normalWS, tangentWS, bitangentWS, optixGetPayload_1(), optixGetPayload_5()))
		{
			optixSetPayload_2(__float_as_uint(radiance.x));
			optixSetPayload_3(__float_as_uint(radiance.y));
			optixSetPayload_4(__float_as_uint(radiance.z));
		}
		else
		{
			// Empty hits are valid too?
			//optixSetPayload_0(0);
		}
	}

	extern "C" __global__ void __closesthit__shadow()
	{
		optixSetPayload_0(1);
	}
}