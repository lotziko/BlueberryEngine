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
			int meshCount = scene->getMeshCount();

			for (int meshId = 0; meshId < meshCount; ++meshId)
			{
				const ofbx::GeometryData& geom = (*scene->getMesh(meshId)).getGeometryData();
				ofbx::Vec3Attributes positions = geom.getPositions();
				ofbx::Vec3Attributes normals = geom.getNormals();
				ofbx::Vec2Attributes uvs = geom.getUVs();

				object = Mesh::Create();
				ObjectDB::AllocateIdToGuid(object, guid, 1);
				object->SetVertices((Vector3*)positions.values, positions.values_count);
				object->SetIndices((UINT*)positions.indices, positions.count);
				object->Apply();

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
						}

						bool hasNormals = normals.values != nullptr;

						bool hasUvs = uvs.values != nullptr;
					}
				}
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