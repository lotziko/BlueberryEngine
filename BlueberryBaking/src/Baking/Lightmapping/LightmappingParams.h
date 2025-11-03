#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include "..\Matrix.h"
#include "..\BVH.h"

namespace Blueberry
{
	#define ACCUMULATION_FRAMES_COUNT 128
	#define BOUNCE_COUNT 4
	#define MATERIAL_COLOR_SIZE 256
	#define SKY_COLOR_SIZE 512

	struct __align__(16) DirectionalLight
	{
		float3 direction;
		float3 color;
	};

	struct __align__(16) PointLight
	{
		float3 position;
		float3 color;
	};

	struct __align__(16) LightmappingParams
	{
		unsigned int accumulationFrameIndex;
		float4* accumulatedImage;

		uint2* validTexels;
		unsigned int validTexelsCount;
		unsigned int* completeCounter;

		float4* color;
		float3* normal;
		float4* position;

		uint2 imageSize;
		int samplePerTexel;
		float texelPerUnit;

		DirectionalLight directionalLight;
		PointLight* pointLights;
		unsigned int pointLightCount;
		uchar4* skyColor;
		BVH bvh;

		Matrix3x4* instanceMatrices;
		OptixTraversableHandle handle;
	};

	struct __align__(16) RayGenData
	{
	};

	struct __align__(16) MissData
	{
	};

	struct __align__(16) HitGroupData
	{
		float3* vertices;
		float3* normals;
		float4* tangents;
		float2* uvs;
		uint3* indices;

		uchar4* materialColor;
	};
}