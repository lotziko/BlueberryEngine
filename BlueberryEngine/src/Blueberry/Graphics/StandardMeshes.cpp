#include "Blueberry\Graphics\StandardMeshes.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Graphics\Mesh.h"

namespace Blueberry
{
	Mesh* StandardMeshes::s_FullscreenMesh = nullptr;
	Mesh* StandardMeshes::s_BlitMesh = nullptr;
	Mesh* StandardMeshes::s_PlaneMesh = nullptr;
	Mesh* StandardMeshes::s_SphereMesh = nullptr;
	Mesh* StandardMeshes::s_CubeMesh = nullptr;
	Mesh* StandardMeshes::s_ConeMesh = nullptr;
	Mesh* StandardMeshes::s_CylinderMesh = nullptr;

	void StandardMeshes::Initialize()
	{
		GetPlane();
	}

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
			Vector3 normals[] =
			{
				{ 0.0f, 0.0f, -1.0f },
				{ 0.0f, 0.0f, -1.0f },
				{ 0.0f, 0.0f, -1.0f },
				{ 0.0f, 0.0f, -1.0f }
			};
			Vector2 uvs[] =
			{
				{ 0.0f, 1.0f }, // fullscreen mesh is flipped vertically to avoid NDC inversion
				{ 0.0f, 0.0f },
				{ 1.0f, 0.0f },
				{ 1.0f, 1.0f }
			};
			uint32_t indices[] = { 0, 1, 2, 2, 3, 0 };
			s_FullscreenMesh = Mesh::Create();
			s_FullscreenMesh->SetName("Fullscreen");
			s_FullscreenMesh->SetVertices(vertices, 4);
			s_FullscreenMesh->SetNormals(normals, 4);
			s_FullscreenMesh->SetIndices(indices, 6);
			s_FullscreenMesh->SetUVs(0, uvs, 4);
			s_FullscreenMesh->GenerateTangents();
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
			Vector3 normals[] =
			{
				{ 0.0f, 1.0f, 0.0f },
				{ 0.0f, 1.0f, 0.0f },
				{ 0.0f, 1.0f, 0.0f },
				{ 0.0f, 1.0f, 0.0f }
			};
			Vector2 uvs[] =
			{
				{ 0.0f, 0.0f },
				{ 0.0f, 1.0f },
				{ 1.0f, 1.0f },
				{ 1.0f, 0.0f }
			};
			uint32_t indices[] = { 0, 1, 2, 2, 3, 0 };
			s_PlaneMesh = Mesh::Create();
			s_PlaneMesh->SetName("Plane");
			s_PlaneMesh->SetVertices(vertices, 4);
			s_PlaneMesh->SetNormals(normals, 4);
			s_PlaneMesh->SetIndices(indices, 6);
			s_PlaneMesh->SetUVs(0, uvs, 4);
			s_PlaneMesh->GenerateTangents();
			s_PlaneMesh->Apply();
			ObjectDB::AllocateIdToGuid(s_PlaneMesh, Guid(TO_HASH("Plane"), 0), 1);
		}

		return s_PlaneMesh;
	}

	Mesh* StandardMeshes::GetSphere()
	{
		if (s_SphereMesh == nullptr)
		{
			float radius = 0.5f;
			int latitudeSegments = 20;
			int longitudeSegments = 20;
			int vertexCount = (latitudeSegments + 1) * (longitudeSegments + 1);
			int indexCount = latitudeSegments * longitudeSegments * 6;

			List<Vector3> vertices(vertexCount);
			List<Vector3> normals(vertexCount);
			List<Vector2> uvs(vertexCount);
			List<uint32_t> indices(indexCount);

			Vector3* verticesPtr = vertices.data();
			Vector3* normalsPtr = normals.data();
			Vector2* uvsPtr = uvs.data();
			uint32_t* indicesPtr = indices.data();

			uint32_t offset = 0;
			for (int lat = 0; lat <= latitudeSegments; ++lat)
			{
				float theta = lat * Math::Pi / latitudeSegments;
				float sinTheta = sin(theta);
				float cosTheta = cos(theta);

				for (int lon = 0; lon <= longitudeSegments; ++lon)
				{
					float phi = lon * 2 * Math::Pi / longitudeSegments;
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
			s_SphereMesh->SetName("Sphere");
			s_SphereMesh->SetVertices(vertices.data(), vertexCount);
			s_SphereMesh->SetNormals(normals.data(), vertexCount);
			s_SphereMesh->SetIndices(indices.data(), indexCount);
			s_SphereMesh->SetUVs(0, uvs.data(), vertexCount);
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
			Vector3 normals[] =
			{
				{ 0.0f, 0.0f, -1.0f },
				{ 0.0f, 0.0f, -1.0f },
				{ 0.0f, 0.0f, -1.0f },
				{ 0.0f, 0.0f, -1.0f },

				{ 0.0f, 0.0f, 1.0f },
				{ 0.0f, 0.0f, 1.0f },
				{ 0.0f, 0.0f, 1.0f },
				{ 0.0f, 0.0f, 1.0f },

				{ 0.0f, 1.0f, 0.0f },
				{ 0.0f, 1.0f, 0.0f },
				{ 0.0f, 1.0f, 0.0f },
				{ 0.0f, 1.0f, 0.0f },

				{ 0.0f, -1.0f, 0.0f },
				{ 0.0f, -1.0f, 0.0f },
				{ 0.0f, -1.0f, 0.0f },
				{ 0.0f, -1.0f, 0.0f },

				{ -1.0f, 0.0f, 0.0f },
				{ -1.0f, 0.0f, 0.0f },
				{ -1.0f, 0.0f, 0.0f },
				{ -1.0f, 0.0f, 0.0f },

				{ 1.0f, 0.0f, 0.0f },
				{ 1.0f, 0.0f, 0.0f },
				{ 1.0f, 0.0f, 0.0f },
				{ 1.0f, 0.0f, 0.0f }
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
				{ 1.0f, 0.0f }
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
			s_CubeMesh->SetName("Cube");
			s_CubeMesh->SetVertices(vertices, ARRAYSIZE(vertices));
			s_CubeMesh->SetNormals(normals, ARRAYSIZE(normals));
			s_CubeMesh->SetIndices(indices, ARRAYSIZE(indices));
			s_CubeMesh->SetUVs(0, uvs, ARRAYSIZE(uvs));
			s_CubeMesh->GenerateTangents();
			s_CubeMesh->Apply();
		}

		return s_CubeMesh;
	}

	Mesh* StandardMeshes::GetCone()
	{
		if (s_ConeMesh == nullptr)
		{
			float radius = 0.5f;
			float height = 1.0f;
			uint32_t segments = 20;
			float halfHeight = height * 0.5f;
			float angleStep = 2.0f * Math::Pi / segments;
			int vertexCount = segments * 3 + 4;
			int indexCount = segments * 6;

			List<Vector3> vertices;
			List<Vector3> normals;
			List<Vector2> uvs;
			List<uint32_t> indices;
			vertices.reserve(vertexCount);
			normals.reserve(vertexCount);
			uvs.reserve(vertexCount);
			indices.reserve(indexCount);

			for (uint32_t i = 0; i <= segments; ++i)
			{
				float angle = i * angleStep;

				float x = cosf(angle);
				float z = sinf(angle);

				Vector3 normal = Vector3(height * x, radius, height * z);
				normal.Normalize();

				float u = static_cast<float>(i) / segments;

				vertices.push_back(Vector3(x * radius, -halfHeight, z * radius));
				normals.push_back(normal);
				uvs.push_back(Vector2(u, 1.0f));

				vertices.push_back(Vector3(0.0f, halfHeight, 0.0f));
				normals.push_back(normal);
				uvs.push_back(Vector2(u, 0.0f));
			}

			for (uint32_t i = 0; i < segments; ++i)
			{
				uint32_t base = i * 2;
				indices.insert(indices.end(), { base, base + 1, base + 2, base + 1, base + 3, base + 2 });
			}

			uint32_t centerIndex = static_cast<uint32_t>(vertices.size());

			vertices.push_back(Vector3(0.0f, -halfHeight, 0.0f));
			normals.push_back(Vector3(0.0f, -1.0f, 0.0f));
			uvs.push_back(Vector2(0.5f, 0.5f));

			uint32_t ringStart = static_cast<uint32_t>(vertices.size());

			for (uint32_t i = 0; i <= segments; ++i)
			{
				float angle = i * angleStep;
				float x = cosf(angle);
				float z = sinf(angle);

				vertices.push_back(Vector3(x * radius, -halfHeight, z * radius));
				normals.push_back(Vector3(0.0f, -1.0f, 0.0f));
				uvs.push_back(Vector2(x * 0.5f + 0.5f, z * 0.5f + 0.5f));
			}

			for (uint32_t i = 0; i < segments; ++i)
			{
				indices.insert(indices.end(), { centerIndex, ringStart + i + 1, ringStart + i });
			}

			s_ConeMesh = Mesh::Create();
			s_ConeMesh->SetName("Cone");
			s_ConeMesh->SetVertices(vertices.data(), vertexCount);
			s_ConeMesh->SetNormals(normals.data(), vertexCount);
			s_ConeMesh->SetIndices(indices.data(), indexCount);
			s_ConeMesh->SetUVs(0, uvs.data(), vertexCount);
			s_ConeMesh->GenerateTangents();
			s_ConeMesh->Apply();
		}

		return s_ConeMesh;
	}

	Mesh* StandardMeshes::GetCylinder()
	{
		if (s_CylinderMesh == nullptr)
		{
			float radius = 0.5f;
			float height = 1.0f;
			uint32_t segments = 20;
			int vertexCount = segments * 4 + 6;
			int indexCount = segments * 12;

			List<Vector3> vertices;
			List<Vector3> normals;
			List<Vector2> uvs;
			List<uint32_t> indices;
			vertices.reserve(vertexCount);
			normals.reserve(vertexCount);
			uvs.reserve(vertexCount);
			indices.reserve(indexCount);

			float halfHeight = height * 0.5f;
			float angleStep = 2.0f * Math::Pi / segments;

			for (uint32_t i = 0; i <= segments; ++i)
			{
				float angle = i * angleStep;

				float x = cosf(angle);
				float z = sinf(angle);

				Vector3 normal(x, 0.0f, z);
				float u = static_cast<float>(i) / segments;

				vertices.push_back(Vector3(x * radius, -halfHeight, z * radius));
				normals.push_back(normal);
				uvs.push_back(Vector2(u, 1.0f));

				vertices.push_back(Vector3(x * radius, halfHeight, z * radius));
				normals.push_back(normal);
				uvs.push_back(Vector2(u, 0.0f));
			}

			for (uint32_t i = 0; i < segments; ++i)
			{
				uint32_t base = i * 2;
				indices.insert(indices.end(), { base, base + 1, base + 2, base + 1, base + 3, base + 2 });
			}

			// Top cap
			{
				float y = halfHeight;
				uint32_t centerIndex = static_cast<uint32_t>(vertices.size());
				Vector3 normal(0.0f, 1.0f, 0.0f);

				vertices.push_back(Vector3(0.0f, y, 0.0f));
				normals.push_back(normal);
				uvs.push_back(Vector2(0.5f, 0.5f));

				uint32_t ringStart = static_cast<uint32_t>(vertices.size());

				for (uint32_t i = 0; i <= segments; ++i)
				{
					float angle = i * angleStep;
					float x = cosf(angle);
					float z = sinf(angle);

					vertices.push_back(Vector3(x * radius, y, z * radius));
					normals.push_back(normal);
					uvs.push_back(Vector2(x * 0.5f + 0.5f, z * 0.5f + 0.5f));
				}

				for (uint32_t i = 0; i < segments; ++i)
				{
					indices.insert(indices.end(), { centerIndex, ringStart + i, ringStart + i + 1 });
				}
			}

			// Bottom cap
			{
				float y = -halfHeight;
				uint32_t centerIndex = static_cast<uint32_t>(vertices.size());
				Vector3 normal(0.0f, -1.0f, 0.0f);

				vertices.push_back(Vector3(0.0f, y, 0.0f));
				normals.push_back(normal);
				uvs.push_back(Vector2(0.5f, 0.5f));

				uint32_t ringStart = static_cast<uint32_t>(vertices.size());

				for (uint32_t i = 0; i <= segments; ++i)
				{
					float angle = i * angleStep;
					float x = cosf(angle);
					float z = sinf(angle);

					vertices.push_back(Vector3(x * radius, y, z * radius));
					normals.push_back(normal);
					uvs.push_back(Vector2(x * 0.5f + 0.5f, z * 0.5f + 0.5f));
				}

				for (uint32_t i = 0; i < segments; ++i)
				{
					indices.insert(indices.end(), { centerIndex, ringStart + i + 1, ringStart + i });
				}
			}

			s_CylinderMesh = Mesh::Create();
			s_CylinderMesh->SetName("Cylinder");
			s_CylinderMesh->SetVertices(vertices.data(), vertexCount);
			s_CylinderMesh->SetNormals(normals.data(), vertexCount);
			s_CylinderMesh->SetIndices(indices.data(), indexCount);
			s_CylinderMesh->SetUVs(0, uvs.data(), vertexCount);
			s_CylinderMesh->GenerateTangents();
			s_CylinderMesh->Apply();
		}

		return s_CylinderMesh;
	}
}