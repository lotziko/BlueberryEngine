#include "bbpch.h"
#include "ModelImporter.h"
#include "Editor\Assets\AssetDB.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
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
		
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			// TODO multiple meshes in file
			// TODO rewrite this, it makes allocateid too and add fileid serializing to yamlserializer
			auto objects = AssetDB::LoadAssetObjects(guid, GetImportedObjects());
			for (auto& pair : objects)
			{
				Object* object = pair.first;
				FileId id = pair.second;
				ObjectDB::AllocateIdToGuid(object, guid, id);
				object->SetState(ObjectState::Default);

				if (object->IsClassType(Mesh::Type))
				{
					(static_cast<Mesh*>(object))->Apply();
					AddImportedObject(object, id);
					BB_INFO("Mesh \"" << object->GetName() << "\" imported from cache.");
				} 
				else if (object->IsClassType(Entity::Type))
				{
					AddImportedObject(object, id);
					BB_INFO("Entity \"" << object->GetName() << "\" imported from cache.");
				}
			}
			return;
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

			std::vector<Object*> objects;

			Transform* root = nullptr;
			if (meshCount > 1)
			{
				Entity* entity = Object::Create<Entity>();
				entity->SetName(GetName());
				root = Object::Create<Transform>();
				entity->AddComponent(root);
				objects.emplace_back(entity);
				size_t entityFileId = TO_HASH(std::string(GetName()).append("_Entity"));
				ObjectDB::AllocateIdToGuid(entity, guid, entityFileId);
				AddImportedObject(entity, entityFileId);
				SetMainObject(entityFileId);
			}

			for (int meshId = 0; meshId < meshCount; ++meshId)
			{
				const ofbx::Mesh* mesh = scene->getMesh(meshId); 
				const ofbx::GeometryData& geom = mesh->getGeometryData();
				ofbx::Vec3Attributes positions = geom.getPositions();
				ofbx::Vec3Attributes normals = geom.getNormals();
				ofbx::Vec2Attributes uvs = geom.getUVs();

				ofbx::DVec3 scale = mesh->getLocalScaling();
				ofbx::DVec3 translation = mesh->getLocalTranslation();
				ofbx::DVec3 rotation = mesh->getLocalRotation();

				Mesh* object = Mesh::Create();
				size_t meshFileId = TO_HASH(mesh->name);
				ObjectDB::AllocateIdToGuid(object, guid, meshFileId);

				Entity* entity = Object::Create<Entity>();
				Transform* transform = Object::Create<Transform>();
				if (root != nullptr)
				{
					transform->SetParent(root);
				}
				transform->SetLocalPosition(Vector3(translation.x / scale.x, translation.y / scale.y, translation.z / scale.z));
				transform->SetLocalRotation(Quaternion::CreateFromYawPitchRoll(ToRadians(rotation.y), ToRadians(rotation.x + 90), ToRadians(rotation.z)));
				MeshRenderer* meshRenderer = Object::Create<MeshRenderer>();
				meshRenderer->SetMesh(object);
				// TODO material
				entity->SetName(mesh->name);
				entity->AddComponent(transform);
				entity->AddComponent(meshRenderer);

				size_t entityFileId = TO_HASH(std::string(mesh->name).append("_Entity"));
				ObjectDB::AllocateIdToGuid(entity, guid, entityFileId);
				AddImportedObject(entity, entityFileId);
				if (root == nullptr)
				{
					SetMainObject(entityFileId);
				}
				objects.emplace_back(entity);

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
								ofbx::Vec2 u = uvs.get(i);
								meshNormals.emplace_back(n);
								meshUVs.emplace_back(u);
							}
						}


						bool hasUvs = uvs.values != nullptr;

						if (polygon.vertex_count == 3)
						{
							meshIndices.emplace_back(polygon.from_vertex);
							meshIndices.emplace_back(polygon.from_vertex + 1);
							meshIndices.emplace_back(polygon.from_vertex + 2);
						}
						else if (polygon.vertex_count == 4)
						{
							meshIndices.emplace_back(polygon.from_vertex);
							meshIndices.emplace_back(polygon.from_vertex + 1);
							meshIndices.emplace_back(polygon.from_vertex + 2);
							meshIndices.emplace_back(polygon.from_vertex);
							meshIndices.emplace_back(polygon.from_vertex + 2);
							meshIndices.emplace_back(polygon.from_vertex + 3);
						}
						else
						{
							// TODO
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

				BB_INFO("Mesh \"" << mesh->name << "\" imported.");
				object->SetName(mesh->name);
				AddImportedObject(object, meshFileId);
				objects.emplace_back(object);
			}

			AssetDB::SaveAssetObjectsToCache(objects);
			scene->destroy();
			delete[] content;
			fclose(fp);
		}
	}
}