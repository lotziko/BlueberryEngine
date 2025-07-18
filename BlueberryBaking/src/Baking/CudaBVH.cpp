#include "BVH.h"
#include "CudaBVH.h"
#include "VecMath.h"

namespace Blueberry
{
	static const uint32_t s_MaxDepth = 8;
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
		const List<Vector2>& vertices = input.vertices;
		const List<uint32_t>& indices = input.indices;

		for (size_t i = 0; i < indices.size(); i += 3)
		{
			Vector2 p1 = vertices[indices[i]];
			Vector2 p2 = vertices[indices[i + 1]];
			Vector2 p3 = vertices[indices[i + 2]];

			BVHTriangle triangle = { make_float2(p1.x, p1.y), make_float2(p2.x, p2.y), make_float2(p3.x, p3.y), make_uint2(0, i) };
			s_Triangles.emplace_back(std::move(triangle));
		}

		BVHInstance instance = { (float3*)input.vertexBuffer, (float3*)input.normalBuffer, (float4*)input.tangentBuffer, (uint3*)input.indexBuffer };
		s_Instances.emplace_back(instance);
	}

	void CUDABVH::Build()
	{
		uint32_t rootIndex = CreateNode();
		BVHNode root = {};
		root.bounds = make_float4(0, 0, 1, 1);
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
		float2 nodeCenter = make_float2((nodeBounds.x + nodeBounds.z) / 2, (nodeBounds.y + nodeBounds.w) / 2);
		uint32_t nodeSplitAxis = nodeSize.x > nodeSize.y ? 0 : 1;
		float nodeSplitPos = getByIndex(nodeCenter, nodeSplitAxis);

		// Use cost too here to avoid empty nodes
		if (depth < s_MaxDepth && triangleCount > 1)
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
		s_Nodes.emplace_back(node);
		return static_cast<uint32_t>(size);
	}
}
