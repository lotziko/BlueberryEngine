#include "bbpch.h"
#include "StandardMeshes.h"

#include "Blueberry\Graphics\Mesh.h"

namespace Blueberry
{
	Mesh* StandardMeshes::GetFullscreen()
	{
		if (s_FullscreenMesh == nullptr)
		{
			Vector3 vertices[] =
			{
				{ -1.0f, -1.0f, 0.0f },
				{ -1.0f, 1.0f, 0.0f },
				{ 1.0f, 1.0f, 0.0f },
				{ 1.0f, -1.0f, 0.0f }
			};
			Vector2 uvs[] =
			{
				{ 0.0f, 1.0f }, // fullscreen meshes are flipped vertically
				{ 0.0f, 0.0f },
				{ 1.0f, 0.0f },
				{ 1.0f, 1.0f },
			};
			UINT indices[] = { 0, 1, 2, 2, 3, 0 };
			s_FullscreenMesh = Mesh::Create();
			s_FullscreenMesh->SetVertices(vertices, 4);
			s_FullscreenMesh->SetIndices(indices, 6);
			s_FullscreenMesh->SetUVs(0, uvs, 4);
			s_FullscreenMesh->Apply();
		}

		return s_FullscreenMesh;
	}

	Mesh* StandardMeshes::GetPlane()
	{
		if (s_PlaneMesh == nullptr)
		{
			Vector3 vertices[] =
			{
				{ -1.0f, 0.0f, -1.0f },
				{ -1.0f, 0.0f, 1.0f },
				{ 1.0f, 0.0f, 1.0f },
				{ 1.0f, 0.0f, -1.0f }
			};
			Vector2 uvs[] =
			{
				{ 0.0f, 0.0f },
				{ 0.0f, 1.0f },
				{ 1.0f, 1.0f },
				{ 1.0f, 0.0f },
			};
			UINT indices[] = { 0, 1, 2, 2, 3, 0 };
			s_PlaneMesh = Mesh::Create();
			s_PlaneMesh->SetVertices(vertices, 4);
			s_PlaneMesh->SetIndices(indices, 6);
			s_PlaneMesh->SetUVs(0, uvs, 4);
			s_PlaneMesh->Apply();
		}

		return s_PlaneMesh;
	}
}