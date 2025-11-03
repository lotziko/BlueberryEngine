#include <optix.h>
#include "LightmappingParams.h"

#include "..\VecMath.h"
#include "..\Random.h"
#include "..\Matrix.h"
#include "Misc.h"

namespace Blueberry
{
	extern "C" 
	{
		__constant__ LightmappingParams params;
	}

	#define MAX_OCCLUSION_RADIUS 1.0f

	struct SamplePositionData
	{
		float3 positionWS;
		float3 normalWS;
		float3 tangentWS;
		float3 bitangentWS;
	};

	struct __align__(16) SampleData
	{
		SamplePositionData samplePositions[64];
		unsigned int validPositionCount;
		float radius;
	};

	//// Based on https://ndotl.wordpress.com/2018/08/29/baking-artifact-free-lightmaps/
	//static __forceinline__ __device__ bool traceBackface(float3& positionWS, float3 normalWS, float3 tangentWS, float3 bitangentWS)
	//{
	//	float maxDistance = (1.0f / params.texelPerUnit) * 0.5f;
	//	float3 rayOrigin = positionWS + normalWS * 0.001f;
	//	float3 rayDirections[4] = { tangentWS, -tangentWS, bitangentWS, -bitangentWS };
	//	for (int i = 0; i < 4; ++i)
	//	{
	//		unsigned int valid = 0;
	//		unsigned int c0 = 0;
	//		unsigned int c1 = 0;
	//		unsigned int c2 = 0;
	//		if (!isValid(rayOrigin))
	//		{
	//			continue;
	//		}
	//		optixTrace(
	//			params.handle,
	//			rayOrigin,
	//			rayDirections[i],
	//			0,											// Min intersection distance
	//			maxDistance,								// Max intersection distance
	//			0.0f,
	//			OptixVisibilityMask(255),
	//			OPTIX_RAY_FLAG_NONE,
	//			0,											// SBT offset
	//			2,											// SBT stride 
	//			0,											// missSBTIndex
	//			valid, c0, c1, c2);
	//		if (valid == 1)
	//		{
	//			float3 newPositionWS = make_float3(__uint_as_float(c0), __uint_as_float(c1), __uint_as_float(c2));
	//			if (isValid(newPositionWS))
	//			{
	//				positionWS = newPositionWS;
	//				return true;
	//			}
	//		}
	//	}
	//	return false;
	//}

	static __forceinline__ __device__ float3 getRandomDirection(float3 normalWS, float3 tangentWS, float3 bitangentWS, unsigned int& seed)
	{
		const float z1 = rnd(seed);
		const float z2 = rnd(seed);

		float3 randomDirection;
		cosineSampleHemisphere(z1, z2, randomDirection);

		return randomDirection.x * tangentWS + randomDirection.y * bitangentWS + randomDirection.z * normalWS;
	}

	static __forceinline__ __device__ bool traceOcclusion(SampleData& data, unsigned int& seed)
	{
		data.radius = MAX_OCCLUSION_RADIUS;

		int invalidCount = 0;
		for (int i = 0; i < 128; ++i)
		{
			SamplePositionData positionData = data.samplePositions[rnd_range(seed, data.validPositionCount)];
			float3 rayOrigin = positionData.positionWS + positionData.normalWS * 0.001f;

			if (!isValid(rayOrigin))
			{
				return false;
			}

			float3 rayDirection = getRandomDirection(positionData.normalWS, positionData.tangentWS, positionData.bitangentWS, seed);

			unsigned int c0 = 0;
			optixTrace(
				params.handle,
				rayOrigin,
				rayDirection,
				0.001f,										// Min intersection distance
				MAX_OCCLUSION_RADIUS,						// Max intersection distance
				0.0f,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				0,											// SBT offset
				3,											// SBT stride 
				0,											// missSBTIndex
				c0);
			float newRadius = __uint_as_float(c0);
			if (newRadius > MAX_OCCLUSION_RADIUS)
			{
				invalidCount += 1;
			}
			data.radius = fmin(newRadius, data.radius);
		}
		return invalidCount < 96;
	}

	static __forceinline__ __device__ bool traceRadiance(float3& radiance, SamplePositionData& positionData, unsigned int& seed)
	{
		float3 rayOrigin = positionData.positionWS + positionData.normalWS * 0.001f;

		if (!isValid(rayOrigin))
		{
			return false;
		}

		float3 rayDirection = getRandomDirection(positionData.normalWS, positionData.tangentWS, positionData.bitangentWS, seed);
		unsigned int depth = 0;
		unsigned int c0 = 0;
		unsigned int c1 = 0;
		unsigned int c2 = 0;
		unsigned int t0 = __float_as_uint(1.0f);
		unsigned int t1 = __float_as_uint(1.0f);
		unsigned int t2 = __float_as_uint(1.0f);
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
			depth, c0, c1, c2, t0, t1, t2, seed);
		if (depth == 100) // invalid ray is stored as depth=100 to fit into 8 registers
		{
			return false;
		}
		radiance += make_float3(__uint_as_float(c0), __uint_as_float(c1), __uint_as_float(c2));
		return true;
	}

	static __forceinline__ __device__ float3 sampleSky(float3 direction)
	{
		if (params.skyColor == nullptr)
		{
			return make_float3(0, 0, 0);
		}

		unsigned int faceID;
		float2 uv;
		float majorAxis;
		if (abs(direction.x) >= abs(direction.y) && abs(direction.x) >= abs(direction.z))
		{
			if (direction.x < 0.0)
			{
				faceID = 1;
				uv = make_float2(-direction.z, direction.y);
			}
			else
			{
				faceID = 0;
				uv = make_float2(direction.z, direction.y);
			}
			majorAxis = abs(direction.x);
		}
		else if (abs(direction.y) >= abs(direction.x) && abs(direction.y) >= abs(direction.z))
		{
			if (direction.y < 0.0)
			{
				faceID = 3;
				uv = make_float2(direction.x, -direction.z);
			}
			else
			{
				faceID = 2;
				uv = make_float2(direction.x, direction.z);
			}
			majorAxis = abs(direction.y);
		}
		else
		{
			if (direction.z < 0.0)
			{
				faceID = 5;
				uv = make_float2(direction.x, direction.y);
			}
			else
			{
				faceID = 4;
				uv = make_float2(-direction.x, direction.y);
			}
			majorAxis = abs(direction.z);
		}
		uv = make_float2(0.5f * (uv.x / majorAxis + 1.0f), 0.5f * (uv.y / majorAxis + 1.0f));
		uv = make_float2((uv.x + faceID) / 6.0f, 1.0 - uv.y);
		uchar4 skyColor = params.skyColor[(unsigned int)(uv.x * (SKY_COLOR_SIZE * 6 - 1)) + (SKY_COLOR_SIZE * 6) * (unsigned int)(uv.y * (SKY_COLOR_SIZE - 1))];
		return make_float3(skyColor.x / 255.0f, skyColor.y / 255.0f, skyColor.z / 255.0f);
	}

	static __forceinline__ __device__  float3 sampleDirectRadiance(float3 positionWS, float3 normalWS)
	{
		float3 rayOrigin = positionWS;
		float3 radiance = {};

		// Directional light
		if (dot(params.directionalLight.color, params.directionalLight.color) > 0)
		{
			float3 rayDirection = params.directionalLight.direction;
			unsigned int isShadow = 0;
			optixTrace(
				params.handle,
				rayOrigin,
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
				radiance += params.directionalLight.color * attenuation;
			}
		}

		// Point lights
		if (params.pointLightCount > 0)
		{
			for (unsigned int i = 0; i < params.pointLightCount; ++i)
			{
				PointLight pointLight = params.pointLights[i];
				float3 posToLight = pointLight.position - positionWS;
				float3 rayDirection = normalize(posToLight);
				float squareDistance = dot(posToLight, posToLight);
				float rayDistance = sqrtf(squareDistance);
				unsigned int isShadow = 0;
				optixTrace(
					params.handle,
					rayOrigin,
					rayDirection,
					0.001f,
					rayDistance,
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
					float NdotL = dot(normalWS, rayDirection);
					float attenuation = clamp(NdotL, 0.0f, 1.0f);
					float falloff = 1.0f / squareDistance;
					radiance += pointLight.color * attenuation * falloff;
				}
			}
		}

		return radiance;
	}

	static __forceinline__ __device__ float3 sampleMaterial(float2 uv, uchar4* color)
	{
		uint2 uvIndex = make_uint2((unsigned int)(uv.x * (MATERIAL_COLOR_SIZE - 1)) % MATERIAL_COLOR_SIZE, (unsigned int)(uv.y * (MATERIAL_COLOR_SIZE - 1)) % MATERIAL_COLOR_SIZE);
		uchar4 materialColor = color[uvIndex.x + MATERIAL_COLOR_SIZE * uvIndex.y];
		return make_float3(materialColor.x / 255.0f, materialColor.y / 255.0f, materialColor.z / 255.0f);
	}

	#define BATCH_SIZE 64
	#define RGB_TO_LUMINANCE make_float3(0.2125f, 0.7154f, 0.0721f)

	extern "C" __global__ void __raygen__pass()
	{
		while (true)
		{
			unsigned int texelIndex = atomicAdd(params.completeCounter, 1);
			
			if (texelIndex >= params.validTexelsCount)
			{
				break;
			}

			const uint2 pos = params.validTexels[texelIndex];

			unsigned int imageIndex = pos.y * params.imageSize.x + pos.x;
			unsigned int seed = tea<4>(imageIndex, params.accumulationFrameIndex);

			float2 texelSize = make_float2(1.0f / params.imageSize.x, 1.0f / params.imageSize.y);

			float3 normalResult = {};
			float3 positionResult = {};

			SampleData sampleData = {};
			for (int i = 0; i < 64; ++i)
			{
				float2 jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
				float2 samplePosition = make_float2((pos.x + 0.5f + jitter.x) * texelSize.x, (pos.y + 0.5f + jitter.y) * texelSize.y);
				//float2 samplePosition = make_float2(pos.x + (i % 8 + 0.5f) / 8.0f, pos.y + (i / 8 + 0.5f) / 8.0f) * texelSize;
				BVHTriangle triangle = params.bvh.GetTriangle(samplePosition);
				if (triangle.index.y != UINT_MAX)
				{
					BVHInstance instance = params.bvh.instances[triangle.index.x];
					Matrix3x4 localToWorld = params.instanceMatrices[instance.globalIndex];

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
					sampleData.samplePositions[sampleData.validPositionCount] = positionData;
					sampleData.validPositionCount += 1;
					normalResult += normalWS;
					positionResult += positionWS;
				}
			}

			// Avoid sampling invalid texels
			if (sampleData.validPositionCount == 0)
			{
				continue;
			}

			/*if (!traceOcclusion(sampleData, seed))
			{
				continue;
			}*/

			float a = 1.0f;
			float b = 0.2f;
			float t = sampleData.radius / MAX_OCCLUSION_RADIUS;
			const int maxSampleCount = params.samplePerTexel;//static_cast<int>(params.samplePerTexel * (a + t * (b - a)));
			const int minSampleCount = maxSampleCount / 20;

			float3 imageResult = {};
			
			float mean = {};
			float meanDistSquared = {};
			int sampleCount = 0;
			int validSampleCount = 0;

			while (true)
			{
				SamplePositionData positionData = sampleData.samplePositions[rnd_range(seed, sampleData.validPositionCount)];
				float3 radiance = {};
				if (traceRadiance(radiance, positionData, seed))
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

			if (validSampleCount * 100 > maxSampleCount) // Skip texels with low valid sample count and interpolate from nearby ones
			{
				params.color[imageIndex] = make_float4(imageResult / max(validSampleCount, 1), 1.0f);
			}
			params.normal[imageIndex] = normalResult / max(sampleData.validPositionCount, 1u);
			params.position[imageIndex] = make_float4(positionResult / max(sampleData.validPositionCount, 1u), sampleData.radius);
		}
	}

	//extern "C" __global__ void __miss__backface()
	//{
	//}

	extern "C" __global__ void __miss__occlusion()
	{
		optixSetPayload_0(__float_as_uint(MAX_OCCLUSION_RADIUS));
	}

	extern "C" __global__ void __miss__radiance()
	{
		float3 radiance = sampleSky(optixGetWorldRayDirection()) * 10;  // TODO exposure property
		optixSetPayload_1(__float_as_uint(radiance.x));
		optixSetPayload_2(__float_as_uint(radiance.y));
		optixSetPayload_3(__float_as_uint(radiance.z));
	}

	extern "C" __global__ void __miss__shadow()
	{
	}

	//extern "C" __global__ void __closesthit__backface()
	//{
	//	if (optixIsBackFaceHit())
	//	{
	//		const float3 hitPos = optixGetWorldRayOrigin() + (optixGetRayTmax() + 0.001f) * optixGetWorldRayDirection();
	//		optixSetPayload_0(1);
	//		optixSetPayload_1(__float_as_uint(hitPos.x));
	//		optixSetPayload_2(__float_as_uint(hitPos.y));
	//		optixSetPayload_3(__float_as_uint(hitPos.z));
	//	}
	//}

	extern "C" __global__ void __closesthit__occlusion()
	{
		float distance = optixGetRayTmax();
		if (optixIsBackFaceHit())
		{
			distance = 10.0f;
		}
		optixSetPayload_0(__float_as_uint(distance));
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

		float3 positionWS = hitPos;
		float3 normalWS, tangentWS, bitangentWS;
		transformOSToWS(params.instanceMatrices[optixGetInstanceId()], normalOS, tangentOS, normalWS, tangentWS, bitangentWS);

		// Avoid backface
		if (dot(normalWS, optixGetWorldRayDirection()) > 0.0f || !isValid(positionWS))
		{
			optixSetPayload_0(100);
			return;
		}

		unsigned int depth = optixGetPayload_0();
		float3 throughput = make_float3(__uint_as_float(optixGetPayload_4()), __uint_as_float(optixGetPayload_5()), __uint_as_float(optixGetPayload_6()));
		float3 newThroughput = throughput;
		unsigned int seed = optixGetPayload_7();
		
		float3 albedo = make_float3(1, 1, 1);
		if (hitGroupData->materialColor != nullptr)
		{
			float2 u0 = hitGroupData->uvs[index.x];
			float2 u1 = hitGroupData->uvs[index.y];
			float2 u2 = hitGroupData->uvs[index.z];
			albedo = sampleMaterial(u0 * uvw.z + u1 * uvw.x + u2 * uvw.y, hitGroupData->materialColor);
			newThroughput *= albedo;
		}

		float3 radiance = sampleDirectRadiance(positionWS, normalWS) * throughput * albedo;

		// Next ray
		if (depth < BOUNCE_COUNT)
		{
			unsigned int newDepth = depth + 1;

			float3 rayOrigin = positionWS + normalWS * 0.001f;
			float3 rayDirection = getRandomDirection(normalWS, tangentWS, bitangentWS, seed);
			unsigned int c0 = 0;
			unsigned int c1 = 0;
			unsigned int c2 = 0;
			unsigned int t0 = __float_as_uint(newThroughput.x);
			unsigned int t1 = __float_as_uint(newThroughput.y);
			unsigned int t2 = __float_as_uint(newThroughput.z);
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
				newDepth, c0, c1, c2, t0, t1, t2, seed);
		
			if (newDepth < 100)
			{
				radiance += make_float3(__uint_as_float(c0), __uint_as_float(c1), __uint_as_float(c2)) * throughput;
				optixSetPayload_1(__float_as_uint(radiance.x));
				optixSetPayload_2(__float_as_uint(radiance.y));
				optixSetPayload_3(__float_as_uint(radiance.z));
				optixSetPayload_7(seed);
			}
		}
	}

	extern "C" __global__ void __closesthit__shadow()
	{
		optixSetPayload_0(1);
	}
}