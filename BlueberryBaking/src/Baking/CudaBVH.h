#pragma once

#include "Blueberry\Core\Base.h"
#include "BVH.h"
#include "CudaBuffer.h"

#include <optix.h>

namespace Blueberry
{
	/*struct BVHTriangle
	{
		float2 p1;
		float2 p2;
		float2 p3;
		unsigned int index;
	};

	struct BVHNode
	{
		float2 boundsMin;
		float2 boundsMax;

		BVHNode* childA;
		BVHNode* childB;

		List<BVHTriangle> triangles;
	};

	class BVH
	{
	public:
		BVH(const List<Vector3>& vertices, const List<uint32_t>& indices);

		unsigned int GetTriangleIndex(const float2& position);

	private:
		void Split(BVHNode* node, const unsigned int& depth);
		void CalculateBounds(BVHNode* node);
		BVHNode* CreateNode();

	private:
		List<BVHNode> m_Nodes;
		BVHNode* m_Root;
	};*/

	// Based on https://github.com/SebLague/Ray-Tracing/blob/main/Assets/Scripts/BVH.cs

	struct CUDABVHInput
	{
		Vector3* vertices;
		uint32_t* indices;

		uint32_t vertexCount;
		uint32_t indexCount;

		CUdeviceptr vertexBuffer;
		CUdeviceptr normalBuffer;
		CUdeviceptr tangentBuffer;
		CUdeviceptr indexBuffer;
	};

	struct CUDABVH
	{
		CUDABVH();

		void AddInstance(const CUDABVHInput& input);
		void Build();

	private:
		void Split(const uint32_t& nodeIndex, const uint32_t& triangleStart, const uint32_t& triangleCount, const uint32_t& depth);
		void Grow(float4& bounds, const BVHTriangle& triangle);
		float4 CalculateBounds(BVHNode* node);

		std::tuple<uint32_t, float, float> ChooseSplit(const uint32_t& nodeIndex, const uint32_t& triangleStart, const uint32_t& triangleCount);
		float EvaluateSplit(const uint32_t& splitAxis, const float& splitPos, const uint32_t& triangleStart, const uint32_t& triangleCount);
		float CalculateNodeCost(const float2& size, const uint32_t& triangleCount);

		uint32_t CreateNode();
		uint32_t CreateNode(const float4& bounds, const uint32_t& triangleStart, const uint32_t& triangleCount);

	public:
		BVH bvh;
		CUDABuffer nodeBuffer;
		CUDABuffer triangleBuffer;
		CUDABuffer instanceBuffer;
	};
}