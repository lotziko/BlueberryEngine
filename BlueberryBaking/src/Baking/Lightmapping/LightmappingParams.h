#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include "..\Matrix.h"
#include "..\BVH.h"

namespace Blueberry
{
	#define ACCUMULATION_FRAMES_COUNT 128
	#define BOUNCE_COUNT 4

	struct __align__(16) DirectionalLight
	{
		float3 direction;
		float3 color;
	};

	struct __align__(16) LightmappingParams
	{
		unsigned int accumulationFrameIndex;
		float4* accumulatedImage;

		uint2* validTexels;
		float4* color;
		float4* normal;
		unsigned int* chartIndex;

		unsigned int offset;
		uint2 imageSize;
		int samplePerTexel;
		float texelPerUnit;

		float3 camEye;
		float3 camU;
		float3 camV;
		float3 camW;

		DirectionalLight directionalLight;
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
		uint3* indices;
	};
}