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
			uint32_t indices[] = { 0, 1, 2, 2, 3, 0 };
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
			uint32_t indices[] = { 0, 1, 2, 2, 3, 0 };
			s_PlaneMesh = Mesh::Create();
			s_PlaneMesh->SetVertices(vertices, 4);
			s_PlaneMesh->SetIndices(indices, 6);
			s_PlaneMesh->SetUVs(0, uvs, 4);
			s_PlaneMesh->Apply();
		}

		return s_PlaneMesh;
	}

	Mesh* StandardMeshes::GetSphere()
	{
		if (s_SphereMesh == nullptr)
		{
			float radius = 0.5f;
			const int latitudeSegments = 20;
			const int longitudeSegments = 20;
			const int vertexCount = (latitudeSegments + 1) * (longitudeSegments + 1);
			const int indexCount = latitudeSegments * longitudeSegments * 6;

			Vector3 vertices[vertexCount];
			Vector3 normals[vertexCount];
			Vector2 uvs[vertexCount];
			uint32_t indices[indexCount];

			Vector3* verticesPtr = vertices;
			Vector3* normalsPtr = normals;
			Vector2* uvsPtr = uvs;
			uint32_t* indicesPtr = indices;

			uint32_t offset = 0;
			for (int lat = 0; lat <= latitudeSegments; ++lat)
			{
				float theta = lat * Pi / latitudeSegments;
				float sinTheta = sin(theta);
				float cosTheta = cos(theta);

				for (int lon = 0; lon <= longitudeSegments; ++lon)
				{
					float phi = lon * 2 * Pi / longitudeSegments;
					float sinPhi = sin(phi);
					float cosPhi = cos(phi);

					Vector3 vertex = Vector3(radius * cosPhi * sinTheta, radius * cosTheta, radius * sinPhi * sinTheta);
					Vector3 normal = Vector3(vertex.x, vertex.y, vertex.z);
					normal.Normalize();

					verticesPtr[0] = vertex;
					normalsPtr[0] = normal;
					uvsPtr[0] = Vector2(static_cast<float>(lon) / longitudeSegments * 2, static_cast<float>(lat) / latitudeSegments);
					++verticesPtr;
					++normalsPtr;
					++uvsPtr;
				}
			}

			for (int lat = 0; lat < latitudeSegments; ++lat) 
			{
				for (int lon = 0; lon < longitudeSegments; ++lon) 
				{
					unsigned int first = lat * (longitudeSegments + 1) + lon;
					unsigned int second = first + longitudeSegments + 1;

					indicesPtr[0] = first;
					indicesPtr[1] = first + 1;
					indicesPtr[2] = second;
					indicesPtr[3] = second;
					indicesPtr[4] = first + 1;
					indicesPtr[5] = second + 1;

					indicesPtr += 6;
				}
			}

			s_SphereMesh = Mesh::Create();
			s_SphereMesh->SetVertices(vertices, vertexCount);
			s_SphereMesh->SetNormals(normals, vertexCount);
			s_SphereMesh->SetIndices(indices, indexCount);
			s_SphereMesh->SetUVs(0, uvs, vertexCount);
			s_SphereMesh->GenerateTangents();
			s_SphereMesh->Apply();
		}

		return s_SphereMesh;
	}

	Mesh* StandardMeshes::GetCube()
	{
		if (s_CubeMesh == nullptr)
		{
			Vector3 vertices[] =
			{
				{ -0.5f, -0.5f, -0.5f },
				{ -0.5f, 0.5f, -0.5f },
				{ 0.5f, 0.5f, -0.5f },
				{ 0.5f, -0.5f, -0.5f },

				{ -0.5f, -0.5f, 0.5f },
				{ 0.5f, -0.5f, 0.5f },
				{ 0.5f,  0.5f, 0.5f },
				{ -0.5f,  0.5f, 0.5f },

				{ -0.5f, 0.5f, -0.5f },
				{ -0.5f, 0.5f, 0.5f },
				{ 0.5f, 0.5f, 0.5f },
				{ 0.5f, 0.5f, -0.5f },

				{ -0.5f, -0.5f, -0.5f },
				{ 0.5f, -0.5f, -0.5f },
				{ 0.5f, -0.5f, 0.5f},
				{ -0.5f, -0.5f, 0.5f },

				{ -0.5f, -0.5f,  0.5f },
				{ -0.5f, 0.5f, 0.5f },
				{ -0.5f, 0.5f, -0.5f },
				{ -0.5f, -0.5f, -0.5f },

				{ 0.5f, -0.5f, -0.5f },
				{ 0.5f,  0.5f, -0.5f },
				{ 0.5f,  0.5f,  0.5f },
				{ 0.5f, -0.5f,  0.5f }
			};
			Vector2 uvs[] =
			{
				{ 0.0f, 0.0f },
				{ 0.0f, 1.0f },
				{ 1.0f, 1.0f },
				{ 1.0f, 0.0f },

				{ 0.0f, 0.0f },
				{ 0.0f, 1.0f },
				{ 1.0f, 1.0f },
				{ 1.0f, 0.0f },

				{ 0.0f, 0.0f },
				{ 0.0f, 1.0f },
				{ 1.0f, 1.0f },
				{ 1.0f, 0.0f },

				{ 0.0f, 0.0f },
				{ 0.0f, 1.0f },
				{ 1.0f, 1.0f },
				{ 1.0f, 0.0f },

				{ 0.0f, 0.0f },
				{ 0.0f, 1.0f },
				{ 1.0f, 1.0f },
				{ 1.0f, 0.0f },

				{ 0.0f, 0.0f },
				{ 0.0f, 1.0f },
				{ 1.0f, 1.0f },
				{ 1.0f, 0.0f },
			};
			uint32_t indices[] = 
			{ 
				0, 1, 2, 0, 2, 3,
				4, 5, 6, 4, 6, 7,
				8, 9, 10, 8, 10, 11,
				12, 13, 14,	12, 14, 15,
				16, 17, 18, 16, 18, 19,
				20, 21, 22, 20, 22, 23
			};
			s_CubeMesh = Mesh::Create();
			s_CubeMesh->SetVertices(vertices, ARRAYSIZE(vertices));
			s_CubeMesh->SetIndices(indices, ARRAYSIZE(indices));
			s_CubeMesh->SetUVs(0, uvs, ARRAYSIZE(uvs));
			s_CubeMesh->Apply();
		}

		return s_CubeMesh;
	}
}