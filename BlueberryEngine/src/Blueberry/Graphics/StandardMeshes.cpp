#include "bbpch.h"
#include "StandardMeshes.h"

#include "Blueberry\Graphics\Mesh.h"

namespace Blueberry
{
	Mesh* StandardMeshes::GetFullscreen()
	{
		if (s_FullscreenMesh == nullptr)
		{
			VertexLayout layout = VertexLayout{}
				.Append(VertexLayout::Position3D)
				.Append(VertexLayout::TextureCoord);

			float vertices[] = {
				-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
				-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
				1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
				1.0f, -1.0f, 0.0f, 1.0f, 0.0f
			};
			UINT indices[] = { 0, 1, 2, 2, 3, 0 };
			s_FullscreenMesh = Mesh::Create(layout, 4, 6);
			s_FullscreenMesh->SetVertexData(vertices, 4);
			s_FullscreenMesh->SetIndexData(indices, 6);
		}
		return s_FullscreenMesh;
	}
	Mesh* StandardMeshes::GetPlane()
	{
		if (s_PlaneMesh == nullptr)
		{
			VertexLayout layout = VertexLayout{}
				.Append(VertexLayout::Position3D)
				.Append(VertexLayout::TextureCoord);

			float vertices[] = {
				-1.0f, 0.0f, -1.0f, 0.0f, 0.0f,
				-1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
				1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
				1.0f, 0.0f, -1.0f, 1.0f, 0.0f
			};
			UINT indices[] = { 0, 1, 2, 2, 3, 0 };
			s_PlaneMesh = Mesh::Create(layout, 4, 6);
			s_PlaneMesh->SetVertexData(vertices, 4);
			s_PlaneMesh->SetIndexData(indices, 6);
		}
		return s_PlaneMesh;
	}
}