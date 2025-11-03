#include "CudaBVH.h"
#include "BVH.h"
#include "VecMath.h"

namespace Blueberry
{
	static const uint32_t s_MaxDepth = 32;
	static List<BVHTriangle> s_Triangles = {};
	static List<BVHNode> s_Nodes = {};
	static List<BVHInstance> s_Instances = {};

	CUDABVH::CUDABVH()
	{
		s_Triangles.clear();
		s_Nodes.clear();
		s_Instances.clear();
	}

	void CUDABVH::AddInstance(const CUDABVHInput& input)
	{
		Vector3* vertices = input.vertices;
		uint32_t* indices = input.indices;
		uint32_t instanceIndex = s_Instances.size();

		for (size_t i = 0; i < input.indexCount; i += 3)
		{
			Vector3 p1 = vertices[indices[i]];
			Vector3 p2 = vertices[indices[i + 1]];
			Vector3 p3 = vertices[indices[i + 2]];

			BVHTriangle triangle = { make_float2(p1.x, p1.y), make_float2(p2.x, p2.y), make_float2(p3.x, p3.y), make_uint2(instanceIndex, i) };
			s_Triangles.push_back(std::move(triangle));
		}

		BVHInstance instance = { (float3*)input.vertexBuffer, (float3*)input.normalBuffer, (float4*)input.tangentBuffer, (uint3*)input.indexBuffer, input.vertexCount, input.globalIndex };
		s_Instances.push_back(std::move(instance));
	}

	void CUDABVH::Build()
	{
		uint32_t rootIndex = CreateNode();
		BVHNode root = {};

		float4 rootBounds = make_float4(FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN);
		for (size_t i = 0; i < s_Triangles.size(); ++i)
		{
			Grow(rootBounds, s_Triangles[i]);
		}
		root.bounds = rootBounds;
		root.triangleStart = 0;
		root.triangleCount = 0;
		s_Nodes[rootIndex] = std::move(root);
		Split(0, 0, static_cast<uint32_t>(s_Triangles.size()), 0);

		triangleBuffer.alloc_and_upload(s_Triangles.data(), s_Triangles.size());
		nodeBuffer.alloc_and_upload(s_Nodes.data(), s_Nodes.size());
		instanceBuffer.alloc_and_upload(s_Instances.data(), s_Instances.size());

		bvh.triangles = (BVHTriangle*)triangleBuffer.data;
		bvh.nodes = (BVHNode*)nodeBuffer.data;
		bvh.instances = (BVHInstance*)instanceBuffer.data;
	}

	void CUDABVH::Split(const uint32_t& nodeIndex, const uint32_t& triangleStart, const uint32_t& triangleCount, const uint32_t& depth)
	{
		BVHNode node = s_Nodes[nodeIndex];
		float4 nodeBounds = node.bounds;
		float2 nodeSize = make_float2(nodeBounds.z - nodeBounds.x, nodeBounds.w - nodeBounds.y);

		float nodeCost = CalculateNodeCost(nodeSize, triangleCount);
		auto split = ChooseSplit(nodeIndex, triangleStart, triangleCount);
		uint32_t nodeSplitAxis = std::get<0>(split);
		float nodeSplitPos = std::get<1>(split);
		float cost = std::get<2>(split);

		if (cost < nodeCost && depth < s_MaxDepth && triangleCount > 1)
		{
			float4 boundsA = make_float4(FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN);
			float4 boundsB = make_float4(FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN);

			uint32_t countOnA = 0;
			for (uint32_t i = triangleStart; i < triangleStart + triangleCount; i++)
			{
				BVHTriangle triangle = s_Triangles[i];
				float2 triangleCenter = (triangle.p1 + triangle.p2 + triangle.p3) / 3;
				float triangleSplitPos = getByIndex(triangleCenter, nodeSplitAxis);
				if (triangleSplitPos < nodeSplitPos)
				{
					Grow(boundsA, triangle);
					BVHTriangle swap = s_Triangles[triangleStart + countOnA];
					s_Triangles[triangleStart + countOnA] = triangle;
					s_Triangles[i] = swap;
					++countOnA;
				}
				else
				{
					Grow(boundsB, triangle);
				}
			}

			uint32_t countOnB = triangleCount - countOnA;
			uint32_t triangleStartA = triangleStart + 0;
			uint32_t triangleStartB = triangleStart + countOnA;

			uint32_t nodeAIndex = CreateNode(boundsA, triangleStartA, 0);
			uint32_t nodeBIndex = CreateNode(boundsB, triangleStartB, 0);

			node.childAIndex = nodeAIndex;
			node.childBIndex = nodeBIndex;

			Split(nodeAIndex, triangleStart, countOnA, depth + 1);
			Split(nodeBIndex, triangleStart + countOnA, countOnB, depth + 1);
		}
		else
		{
			node.triangleStart = triangleStart;
			node.triangleCount = triangleCount;
		}
		s_Nodes[nodeIndex] = std::move(node);
	}

	void CUDABVH::Grow(float4& bounds, const BVHTriangle& triangle)
	{
		bounds.x = std::min(bounds.x, std::min(triangle.p1.x, std::min(triangle.p2.x, triangle.p3.x)));
		bounds.y = std::min(bounds.y, std::min(triangle.p1.y, std::min(triangle.p2.y, triangle.p3.y)));
		bounds.z = std::max(bounds.z, std::max(triangle.p1.x, std::max(triangle.p2.x, triangle.p3.x)));
		bounds.w = std::max(bounds.w, std::max(triangle.p1.y, std::max(triangle.p2.y, triangle.p3.y)));
	}

	float4 CUDABVH::CalculateBounds(BVHNode* node)
	{
		float4 bounds = make_float4(FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN);
		BVHTriangle* ptr = &s_Triangles[node->triangleStart];
		BVHTriangle* end = ptr + node->triangleCount;
		for (; ptr != end; ++ptr)
		{
			BVHTriangle triangle = *ptr;
			Grow(bounds, triangle);
		}
		return bounds;
	}

	std::tuple<uint32_t, float, float> CUDABVH::ChooseSplit(const uint32_t& nodeIndex, const uint32_t& triangleStart, const uint32_t& triangleCount)
	{
		if (triangleCount <= 1)
		{
			return std::make_tuple(0, 0, FLT_MAX);
		}
		
		float bestSplitPos = 0;
		int bestSplitAxis = 0;
		const uint32_t numSplitTests = 5;
		float bestCost = FLT_MAX;
		BVHNode node = s_Nodes[nodeIndex];
		float2 boundsMin = make_float2(node.bounds.x, node.bounds.y);
		float2 boundsMax = make_float2(node.bounds.z, node.bounds.w);

		for (uint32_t axis = 0; axis < 2; ++axis)
		{
			for (uint32_t i = 0; i < numSplitTests; ++i)
			{
				float splitT = (i + 1) / (numSplitTests + 1.0f);
				float splitPos = lerp(getByIndex(boundsMin, axis), getByIndex(boundsMax, axis), splitT);
				float cost = EvaluateSplit(axis, splitPos, triangleStart, triangleCount);
				if (cost < bestCost)
				{
					bestCost = cost;
					bestSplitPos = splitPos;
					bestSplitAxis = axis;
				}
			}
		}

		return std::make_tuple(bestSplitAxis, bestSplitPos, bestCost);
	}

	float CUDABVH::EvaluateSplit(const uint32_t& splitAxis, const float& splitPos, const uint32_t& triangleStart, const uint32_t& triangleCount)
	{
		float4 boundsA = make_float4(FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN);
		float4 boundsB = make_float4(FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN);

		uint32_t numOnA = 0;
		uint32_t numOnB = 0;

		for (uint32_t i = triangleStart; i < triangleStart + triangleCount; ++i)
		{
			BVHTriangle triangle = s_Triangles[i];
			float2 triangleCenter = (triangle.p1 + triangle.p2 + triangle.p3) / 3;
			if (getByIndex(triangleCenter, splitAxis) < splitPos)
			{
				Grow(boundsA, triangle);
				++numOnA;
			}
			else
			{
				Grow(boundsB, triangle);
				++numOnB;
			}
		}

		float2 sizeA = make_float2(boundsA.z - boundsA.x, boundsA.w - boundsA.y);
		float2 sizeB = make_float2(boundsB.z - boundsB.x, boundsB.w - boundsB.y);

		float costA = CalculateNodeCost(sizeA, numOnA);
		float costB = CalculateNodeCost(sizeB, numOnB);
		return costA + costB;
	}

	float CUDABVH::CalculateNodeCost(const float2& size, const uint32_t& triangleCount)
	{
		return size.x * size.y * triangleCount;
	}

	uint32_t CUDABVH::CreateNode()
	{
		size_t size = s_Nodes.size();
		s_Nodes.emplace_back();
		return static_cast<uint32_t>(size);
	}

	uint32_t CUDABVH::CreateNode(const float4& bounds, const uint32_t& triangleStart, const uint32_t& triangleCount)
	{
		size_t size = s_Nodes.size();
		BVHNode node = {};
		node.bounds = bounds;
		node.triangleStart = triangleStart;
		node.triangleCount = triangleCount;
		s_Nodes.push_back(std::move(node));
		return static_cast<uint32_t>(size);
	}
}
