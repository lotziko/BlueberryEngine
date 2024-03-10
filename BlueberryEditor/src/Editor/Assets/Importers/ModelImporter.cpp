#include "bbpch.h"
#include "ModelImporter.h"
#include "Editor\Assets\AssetDB.h"
#include "Blueberry\Graphics\Mesh.h"
#include "openfbx\ofbx.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AssetImporter, ModelImporter)

	void ModelImporter::BindProperties()
	{

	}

	void ModelImporter::ImportData()
	{
		Guid guid = GetGuid();

		Mesh* object;
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			object = nullptr;
		}
		else
		{
			FILE* fp = fopen(GetFilePath().c_str(), "rb");
			fseek(fp, 0, SEEK_END);
			long fileSize = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			auto* content = new ofbx::u8[fileSize];
			fread(content, 1, fileSize, fp);

			ofbx::LoadFlags flags =
				ofbx::LoadFlags::IGNORE_BLEND_SHAPES |
				ofbx::LoadFlags::IGNORE_CAMERAS |
				ofbx::LoadFlags::IGNORE_LIGHTS |
				ofbx::LoadFlags::IGNORE_SKIN |
				ofbx::LoadFlags::IGNORE_BONES |
				ofbx::LoadFlags::IGNORE_PIVOTS |
				ofbx::LoadFlags::IGNORE_POSES |
				ofbx::LoadFlags::IGNORE_VIDEOS |
				ofbx::LoadFlags::IGNORE_LIMBS |
				ofbx::LoadFlags::IGNORE_ANIMATIONS;

			ofbx::IScene* scene = ofbx::load((ofbx::u8*)content, fileSize, (ofbx::u16)flags);
			int indicesOffset = 0;
			int meshCount = scene->getMeshCount();

			for (int meshId = 0; meshId < meshCount; ++meshId)
			{
				const ofbx::GeometryData& geom = (*scene->getMesh(meshId)).getGeometryData();
				ofbx::Vec3Attributes positions = geom.getPositions();
				ofbx::Vec3Attributes normals = geom.getNormals();
				ofbx::Vec2Attributes uvs = geom.getUVs();

				object = Mesh::Create();
				ObjectDB::AllocateIdToGuid(object, guid, 1);

				std::vector<ofbx::Vec3> meshPositions;
				std::vector<ofbx::Vec3> meshNormals;
				std::vector<ofbx::Vec2> meshUVs;
				std::vector<UINT> meshIndices;

				// Each material on the mesh has a partition (Unity submesh)
				for (int partitionId = 0; partitionId < geom.getPartitionCount(); ++partitionId)
				{
					const ofbx::GeometryPartition& partition = geom.getPartition(partitionId);
					for (int polygonId = 0; polygonId < partition.polygon_count; ++polygonId)
					{
						const ofbx::GeometryPartition::Polygon& polygon = partition.polygons[polygonId];

						for (int i = polygon.from_vertex; i < polygon.from_vertex + polygon.vertex_count; ++i) 
						{
							ofbx::Vec3 v = positions.get(i);
							meshPositions.emplace_back(v);
						}

						bool hasNormals = normals.values != nullptr;
						if (hasNormals) {
							// normals.indices might be different than positions.indices
							// but normals.get(i) is normal for positions.get(i)
							for (int i = polygon.from_vertex; i < polygon.from_vertex + polygon.vertex_count; ++i) 
							{
								ofbx::Vec3 n = normals.get(i);
								meshNormals.emplace_back(n);
							}
						}


						bool hasUvs = uvs.values != nullptr;

						for (int i = polygon.from_vertex; i < polygon.from_vertex + polygon.vertex_count; ++i) 
						{
							int index = /*1 + */i + indicesOffset;
							meshIndices.emplace_back(index);
						}
					}
				}

				// Rotate x axis
				Vector3* verticesPtr = (Vector3*)meshPositions.data();
				Vector3* normalsPtr = (Vector3*)meshNormals.data();
				Matrix rotationMatrix = Matrix::CreateRotationX(ToRadians(-90));
				Vector4 point = Vector4(0, 0, 0, 1);
				Vector4 direction = Vector4::Zero;
				for (int i = 0; i < meshPositions.size(); i++)
				{
					memcpy(&point, verticesPtr, sizeof(float) * 3);
					point = Vector4::Transform(point, rotationMatrix);
					memcpy(verticesPtr, &point, sizeof(float) * 3);

					memcpy(&direction, normalsPtr, sizeof(float) * 3);
					direction = Vector4::Transform(direction, rotationMatrix);
					memcpy(normalsPtr, &direction, sizeof(float) * 3);

					++verticesPtr;
					++normalsPtr;
				}

				object->SetVertices((Vector3*)meshPositions.data(), meshPositions.size());
				object->SetNormals((Vector3*)meshNormals.data(), meshNormals.size());
				object->SetUVs(0, (Vector2*)meshUVs.data(), meshUVs.size());
				object->SetIndices((UINT*)meshIndices.data(), meshIndices.size());
				object->Apply();
			}

			scene->destroy();
			delete[] content;
			fclose(fp);
			BB_INFO("Mesh \"" << GetName() << "\" imported.");
		}
		object->SetName(GetName());
		AddImportedObject(object, 1);
	}
}