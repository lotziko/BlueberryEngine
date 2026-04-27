#pragma once

#include "Blueberry\Core\Base.h"

#ifdef JPH_DEBUG_RENDERER
#include <Jolt\Jolt.h>
#include <Jolt\Renderer\DebugRenderer.h>

namespace Blueberry
{
    class GfxBuffer;

    class PhysicsBatch : public JPH::RefTargetVirtual
    {
    public:
        JPH_OVERRIDE_NEW_DELETE

        PhysicsBatch(const JPH::DebugRenderer::Vertex* inVertices, int inVertexCount, const JPH::uint32* inIndices, int inIndexCount);
        PhysicsBatch(const JPH::DebugRenderer::Triangle* inTriangles, int inTriangleCount);
        ~PhysicsBatch() override;

        void AddRef()
        {
            m_RefCount++;
        }

        void Release()
        {
            m_RefCount--;
            if (m_RefCount == 0)
            {
                delete this;
            }
        }

        void Draw(JPH::ColorArg inColor);

    protected:
        int m_RefCount = 0;
        GfxBuffer* m_Vertices = nullptr;
        GfxBuffer* m_Indices = nullptr;
    };

	class PhysicsGizmoRenderer : public JPH::DebugRenderer
	{
	public:
		JPH_OVERRIDE_NEW_DELETE

        PhysicsGizmoRenderer();
		~PhysicsGizmoRenderer() override = default;

		void DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor) override;
		void DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor, ECastShadow inCastShadow) override;
		void DrawText3D(JPH::RVec3Arg inPosition, const std::string_view& inString, JPH::ColorArg inColor, float inHeight) override;
		JPH::DebugRenderer::Batch CreateTriangleBatch(const JPH::DebugRenderer::Vertex* inVertices, int inVertexCount, const JPH::uint32* inIndices, int inIndexCount) override;
		JPH::DebugRenderer::Batch CreateTriangleBatch(const JPH::DebugRenderer::Triangle* inTriangles, int inTriangleCount) override;
		void DrawGeometry(JPH::RMat44Arg inModelMatrix, const JPH::AABox& inWorldSpaceBounds, float inLODScaleSq, JPH::ColorArg inModelColor, const JPH::DebugRenderer::GeometryRef& inGeometry, JPH::DebugRenderer::ECullMode inCullMode = JPH::DebugRenderer::ECullMode::CullBackFace, JPH::DebugRenderer::ECastShadow inCastShadow = JPH::DebugRenderer::ECastShadow::On, JPH::DebugRenderer::EDrawMode inDrawMode = JPH::DebugRenderer::EDrawMode::Solid) override;
	
        static void Draw();
    };
}
#endif