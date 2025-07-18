#pragma once

#include "VecMath.h"

struct __align__(16) BVHTriangle
{
	float2 p1;
	float2 p2;
	float2 p3;
	uint2 index; // x is instance index, y is triangle index
};

struct __align__(16) BVHInstance
{
	float3* vertices;
	float3* normals;
	float4* tangents;
	uint3* indices;
};

struct __align__(16) BVHNode
{
	float4 bounds;

	unsigned int childAIndex;
	unsigned int childBIndex;

	unsigned int triangleStart;
	unsigned int triangleCount;
};

inline __device__ float Sign(const float2& p1, const float2& p2, const float2& p3)
{
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

inline __device__ bool IsInsideTriangle(float2 position, float2 p1, float2 p2, float2 p3)
{
	bool b1 = Sign(position, p1, p2) < 0.0f;
	bool b2 = Sign(position, p2, p3) < 0.0f;
	bool b3 = Sign(position, p3, p1) < 0.0f;

	return (b1 == b2) && (b2 == b3);
}

//inline __device__ bool IsInsideTriangle(float2 position, float2 p1, float2 p2, float2 p3)
//{
//	float2 v1 = p3 - p1;
//	float2 v2 = p2 - p1;
//	float2 v3 = position - p1;
//
//	float dot11 = dot(v1, v1);
//	float dot12 = dot(v1, v2);
//	float dot13 = dot(v1, v3);
//	float dot22 = dot(v2, v2);
//	float dot23 = dot(v2, v3);
//
//	float denom = dot11 * dot22 - dot12 * dot12;
//	if (denom == 0.0f)
//	{
//		return false;
//	}
//
//	float invDenom = 1.0f / denom;
//	float u = (dot22 * dot13 - dot12 * dot23) * invDenom;
//	float v = (dot11 * dot23 - dot12 * dot13) * invDenom;
//
//	return (u >= 0.0f) && (v >= 0.0f) && (u + v <= 1.0f);
//}

inline __device__ bool IsInsideBounds(float2 position, float4 bounds)
{
	return position.x >= bounds.x && position.y >= bounds.y && position.x <= bounds.z && position.y <= bounds.w;
}

inline __device__  float3 GetBarycentrics(float2 position, float2 p1, float2 p2, float2 p3)
{
	float2 v1 = p3 - p1;
	float2 v2 = p2 - p1;
	float2 v3 = position - p1;

	float dot11 = dot(v1, v1);
	float dot12 = dot(v1, v2);
	float dot13 = dot(v1, v3);
	float dot22 = dot(v2, v2);
	float dot23 = dot(v2, v3);

	float denom = dot11 * dot22 - dot12 * dot12;
	if (denom == 0.0f)
	{
		return {};
	}

	float invDenom = 1.0f / denom;
	float u = (dot11 * dot23 - dot12 * dot13) * invDenom, v = (dot22 * dot13 - dot12 * dot23) * invDenom;
	return make_float3(u, v, (1 - u - v));
}

struct __align__(16) BVH
{
public:
	__device__ BVHTriangle GetTriangle(float2 position)
	{
		unsigned int stack[32];
		unsigned int stackIndex = 0;
		stack[stackIndex++] = 0;
		while (stackIndex > 0)
		{
			BVHNode node = nodes[stack[--stackIndex]];
			bool isLeaf = node.triangleCount > 0;

			if (isLeaf)
			{
				for (unsigned int i = 0; i < node.triangleCount; ++i)
				{
					BVHTriangle triangle = triangles[node.triangleStart + i];
					if (IsInsideTriangle(position, triangle.p1, triangle.p2, triangle.p3))
					{
						return triangle;
					}
				}
			}
			else
			{
				// TODO check closest of two
				BVHNode childA = nodes[node.childAIndex];
				BVHNode childB = nodes[node.childBIndex];

				if (IsInsideBounds(position, childA.bounds))
				{
					stack[stackIndex++] = node.childAIndex;
				}
				if (IsInsideBounds(position, childB.bounds))
				{
					stack[stackIndex++] = node.childBIndex;
				}
			}
		}
		BVHTriangle triangle = {};
		triangle.index = { UINT_MAX, UINT_MAX };
		return triangle;
	}

	BVHNode* nodes;
	BVHTriangle* triangles;
	BVHInstance* instances;
};