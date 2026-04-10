#include "PhysicsGizmoRenderer.h"

#include "Gizmos.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Physics\Physics.h"

#ifdef JPH_DEBUG_RENDERER
#include <Jolt\Physics\PhysicsSystem.h>

namespace Blueberry
{
	static PhysicsGizmoRenderer* s_Renderer = nullptr;

	PhysicsBatch::PhysicsBatch(const JPH::DebugRenderer::Vertex* inVertices, int inVertexCount, const JPH::uint32* inIndices, int inIndexCount)
	{
		List<Vector3> vertices(inVertexCount);
		for (size_t i = 0; i < inVertexCount; ++i)
		{
			JPH::Float3 position = inVertices[i].mPosition;
			vertices[i] = Vector3(position.x, position.y, position.z);
		}

		BufferProperties vertexBufferProperties = {};
		vertexBufferProperties.elementCount = inVertexCount;
		vertexBufferProperties.elementSize = sizeof(Vector3);
		vertexBufferProperties.usageFlags = BufferUsageFlags::VertexBuffer;
		vertexBufferProperties.data = vertices.data();
		vertexBufferProperties.dataSize = sizeof(Vector3) * inVertexCount;

		GfxDevice::CreateBuffer(vertexBufferProperties, m_Vertices);

		BufferProperties indexBufferProperties = {};
		indexBufferProperties.elementCount = inIndexCount;
		indexBufferProperties.elementSize = sizeof(uint32_t);
		indexBufferProperties.usageFlags = BufferUsageFlags::IndexBuffer;
		indexBufferProperties.data = (void*)inIndices;
		indexBufferProperties.dataSize = sizeof(uint32_t) * inIndexCount;

		GfxDevice::CreateBuffer(indexBufferProperties, m_Indices);
	}

	PhysicsBatch::PhysicsBatch(const JPH::DebugRenderer::Triangle* inTriangles, int inTriangleCount)
	{
		List<Vector3> vertices(inTriangleCount * 3);
		for (size_t i = 0; i < inTriangleCount; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				JPH::Float3 position = inTriangles[i].mV[j].mPosition;
				vertices[i * 3 + j] = Vector3(position.x, position.y, position.z);
			}
		}

		BufferProperties vertexBufferProperties = {};
		vertexBufferProperties.elementCount = vertices.size();
		vertexBufferProperties.elementSize = sizeof(Vector3);
		vertexBufferProperties.usageFlags = BufferUsageFlags::VertexBuffer;
		vertexBufferProperties.data = vertices.data();
		vertexBufferProperties.dataSize = sizeof(Vector3) * vertices.size();

		GfxDevice::CreateBuffer(vertexBufferProperties, m_Vertices);
	}

	PhysicsBatch::~PhysicsBatch()
	{
		delete m_Vertices;
		delete m_Indices;
	}

	void PhysicsBatch::Draw(JPH::ColorArg inColor)
	{
		Gizmos::DrawMesh(m_Vertices, m_Indices);
	}

	PhysicsGizmoRenderer::PhysicsGizmoRenderer()
	{
		Initialize();
	}

	void PhysicsGizmoRenderer::DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor)
	{
		JPH::Vec4 col = inColor.ToVec4();
		Gizmos::SetColor(Color(col.GetX(), col.GetY(), col.GetZ(), col.GetW()));
		Gizmos::DrawLine(Vector3(inFrom.GetX(), inFrom.GetY(), inFrom.GetZ()), Vector3(inTo.GetX(), inTo.GetY(), inTo.GetZ()));
	}

	void PhysicsGizmoRenderer::DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor, ECastShadow inCastShadow)
	{
		JPH::Vec4 col = inColor.ToVec4();
		Gizmos::SetColor(Color(col.GetX(), col.GetY(), col.GetZ(), col.GetW()));
		Gizmos::DrawLine(Vector3(inV1.GetX(), inV1.GetY(), inV1.GetZ()), Vector3(inV2.GetX(), inV2.GetY(), inV2.GetZ()));
		Gizmos::DrawLine(Vector3(inV2.GetX(), inV2.GetY(), inV2.GetZ()), Vector3(inV3.GetX(), inV3.GetY(), inV3.GetZ()));
		Gizmos::DrawLine(Vector3(inV3.GetX(), inV3.GetY(), inV3.GetZ()), Vector3(inV1.GetX(), inV1.GetY(), inV1.GetZ()));
	}

	void PhysicsGizmoRenderer::DrawText3D(JPH::RVec3Arg inPosition, const std::string_view& inString, JPH::ColorArg inColor, float inHeight)
	{
	}

	JPH::DebugRenderer::Batch PhysicsGizmoRenderer::CreateTriangleBatch(const JPH::DebugRenderer::Vertex* inVertices, int inVertexCount, const JPH::uint32* inIndices, int inIndexCount)
	{
		return new PhysicsBatch(inVertices, inVertexCount, inIndices, inIndexCount);
	}

	JPH::DebugRenderer::Batch PhysicsGizmoRenderer::CreateTriangleBatch(const JPH::DebugRenderer::Triangle* inTriangles, int inTriangleCount)
	{
		return new PhysicsBatch(inTriangles, inTriangleCount);
	}

	void PhysicsGizmoRenderer::DrawGeometry(JPH::RMat44Arg inModelMatrix, const JPH::AABox& inWorldSpaceBounds, float inLODScaleSq, JPH::ColorArg inModelColor, const JPH::DebugRenderer::GeometryRef& inGeometry, JPH::DebugRenderer::ECullMode inCullMode, JPH::DebugRenderer::ECastShadow inCastShadow, JPH::DebugRenderer::EDrawMode inDrawMode)
	{
		Matrix matrix;
		inModelMatrix.StoreFloat4x4(reinterpret_cast<JPH::Float4*>(matrix.m));
		Gizmos::SetMatrix(matrix);

		const JPH::DebugRenderer::LOD& lod = inGeometry->GetLOD({}, inWorldSpaceBounds, inLODScaleSq);

		PhysicsBatch* batch = reinterpret_cast<PhysicsBatch*>(lod.mTriangleBatch.GetPtr());
		batch->Draw(inModelColor);
	}

	void PhysicsGizmoRenderer::Draw()
	{
		if (Physics::s_PhysicsSystem != nullptr)
		{
			if (s_Renderer == nullptr)
			{
				s_Renderer = new PhysicsGizmoRenderer();
			}

			JPH::BodyManager::DrawSettings drawSettings;
			drawSettings.mDrawShape = true;
			drawSettings.mDrawShapeWireframe = true;
			drawSettings.mDrawBoundingBox = false;
			Physics::s_PhysicsSystem->DrawBodies(drawSettings, s_Renderer);
		}
	}
}
#endif