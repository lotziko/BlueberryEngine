
#include "..\VecMath.h"
#include "DenoisingParams.h"
#include <cuda_runtime.h>

namespace Blueberry
{
	extern "C"
	{
		__constant__ DenoisingParams params;
	}

	static __forceinline__ __device__ float l2Sq(const float4 a, const float4 b)
	{
		float3 d = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
		return d.x*d.x + d.y*d.y + d.z*d.z;
	}

	static __forceinline__ __device__ float safeExpNeg(float x_over_2sigma2)
	{
		x_over_2sigma2 = fminf(fmaxf(x_over_2sigma2, -50.f), 50.f);
		return __expf(-x_over_2sigma2);
	}

	static __forceinline__ __device__ float gaussianSpatial(int dx, int dy, float sigmaSpatial)
	{
		const float r2 = float(dx*dx + dy * dy);
		const float denom = 2.0f * sigmaSpatial * sigmaSpatial + 1e-8f;
		return safeExpNeg(r2 / denom);
	}

	static __forceinline__ __device__ float gaussianRange(const float4 c0, const float4 c1, float sigmaRange)
	{
		const float d2 = l2Sq(c0, c1);
		const float denom = 2.0f * sigmaRange * sigmaRange + 1e-12f;
		return safeExpNeg(d2 / denom);
	}

	static __forceinline__ __device__ float gaussianNormal(const float4 n0, const float4 n1, float sigmaNormal)
	{
		float3 d = make_float3(n0.x - n1.x, n0.y - n1.y, n0.z - n1.z);
		const float d2 = d.x*d.x + d.y*d.y + d.z*d.z;
		const float denom = 2.0f * sigmaNormal * sigmaNormal + 1e-12f;
		return safeExpNeg(d2 / denom);
	}

	extern "C" __global__ void __denoise()
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		
		unsigned int index = y * params.imageSize.x + x;
		const float4 c1 = params.inputColor[index];
		const float4 n1 = params.inputNormal[index];
		const unsigned int chartIndex = params.chartIndex[index];
		const int r = 5;
		const float sigmaSpatial = r * 0.5f;
		const float sigmaRange = 0.05;
		const float sigmaNormal = 0.25;
		const unsigned int size = params.imageSize.x * params.imageSize.y;

		float4 sumCol = {};
		float sumW = 0.0f;

		for (int dy = -r; dy <= r; ++dy)
		{
			int ny = y + dy;
			for (int dx = -r; dx <= r; ++dx)
			{
				unsigned int nearbyIndex = (y + dy) * params.imageSize.x + (x + dx);
				if (nearbyIndex < 0 || nearbyIndex >= size || params.chartIndex[nearbyIndex] != chartIndex)
				{
					continue;
				}
				const float4 c2 = params.inputColor[nearbyIndex];
				float w = gaussianSpatial(dx, dy, sigmaSpatial);
				w *= gaussianRange(c1, c2, sigmaRange);

				const float4 n2 = params.inputNormal[nearbyIndex];
				w *= gaussianNormal(n1, n2, sigmaNormal);

				sumCol.x += w * c2.x;
				sumCol.y += w * c2.y;
				sumCol.z += w * c2.z;
				sumCol.w += w * c2.w;
				sumW += w;
			}
		}

		if (sumW > 0.0f)
		{
			const float invW = 1.0f / sumW;
			params.outputColor[index] = make_float4(sumCol.x * invW, sumCol.y * invW, sumCol.z * invW, sumCol.w * invW);
		}
		else
		{
			params.outputColor[index] = c1;
		}
	}
}