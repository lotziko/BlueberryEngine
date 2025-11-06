#pragma once

#include <cuda_runtime.h>

template<unsigned int N>
__forceinline__ __device__ unsigned int tea(unsigned int val0, unsigned int val1)
{
	unsigned int v0 = val0;
	unsigned int v1 = val1;
	unsigned int s0 = 0;

	for (unsigned int n = 0; n < N; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}

	return v0;
}

// Generate random unsigned int in [0, 2^24)
__forceinline__ __device__ unsigned int lcg(unsigned int &prev)
{
	const unsigned int LCG_A = 1664525u;
	const unsigned int LCG_C = 1013904223u;
	prev = (LCG_A * prev + LCG_C);
	return prev & 0x00FFFFFF;
}

__forceinline__ __device__ unsigned int lcg2(unsigned int &prev)
{
	prev = (prev * 8121 + 28411) % 134456;
	return prev;
}

// Generate random float in [0, 1)
__forceinline__ __device__ float rnd(unsigned int &prev)
{
	return ((float)lcg(prev) / (float)0x01000000);
}

// Generate random unsigned int in [0, max)
__forceinline__ __device__ unsigned int rnd_range(unsigned int &prev, unsigned int max)
{
	return lcg(prev) % max;
}

__forceinline__ __device__ unsigned int rot_seed(unsigned int seed, unsigned int frame)
{
	return seed ^ frame;
}

// The same algorithm used in specular convolution
static __forceinline__ __device__ float RadicalInverse_VdC(unsigned int bits)
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

static __forceinline__ __device__ float2 Hammersley(unsigned int i, unsigned int N)
{
	return make_float2(float(i) / float(N), RadicalInverse_VdC(i));
}

static __forceinline__ __device__ float3 CosineSampleHemisphere(float2 u)
{
	float r = sqrt(u.x);
	float theta = 2.0f * M_PIf * u.y;

	float x = r * cos(theta);
	float y = r * sin(theta);
	float z = sqrt(1.0f - u.x); // Ensures cosine-weighted

	return make_float3(x, y, z);
}