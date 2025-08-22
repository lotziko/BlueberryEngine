#pragma once

#include <cuda_runtime.h>

namespace Blueberry
{
	struct __align__(16) DenoisingParams
	{
		uint2 imageSize;

		float4* inputColor;
		float4* inputNormal;
		float4* outputColor;
		unsigned int* chartIndex;
	};
}