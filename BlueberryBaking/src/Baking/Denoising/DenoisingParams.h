#pragma once

#include <cuda_runtime.h>

namespace Blueberry
{
	struct __align__(16) DenoisingParams
	{
		uint2 imageSize;

		float4* inputColor;
		float3* inputNormal;
		float4* inputPosition;
		float4* outputColor;
		unsigned int* chartIndex;
	};
}