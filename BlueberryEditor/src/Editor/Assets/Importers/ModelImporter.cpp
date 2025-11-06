#include "ModelImporter.h"

#include "Editor\Assets\AssetDB.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Physics\PhysicsShapeCache.h"
#include "Blueberry\Tools\FileHelper.h"

#include <fbxsdk.h>
#include <xatlas\xatlas.h>
#include <fstream>

namespace Blueberry
{
	DATA_DEFINITION(ModelMaterialData)
	{
		DEFINE_FIELD(ModelMaterialData, m_Name, BindingType::String, {})
		DEFINE_FIELD(ModelMaterialData, m_Material, BindingType::ObjectPtr, FieldOptions().SetObjectType(Material::Type))
	}

	OBJECT_DEFINITION(ModelImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(ModelImporter, AssetImporter)
		DEFINE_FIELD(ModelImporter, m_Materials, BindingType::DataList, FieldOptions().SetObjectType(ModelMaterialData::Type))
		DEFINE_FIELD(ModelImporter, m_Scale, BindingType::Float, {})
		DEFINE_FIELD(ModelImporter, m_GenerateLightmapUV, BindingType::Bool, {})
		DEFINE_FIELD(ModelImporter, m_GeneratePhysicsShape, BindingType::Bool, {})
	}

	const String& ModelMaterialData::GetName()
	{
		return m_Name;
	}

	void ModelMaterialData::SetName(const String& name)
	{
		m_Name = name;
	}

	Material* ModelMaterialData::GetMaterial()
	{
		return m_Material.Get();
	}

	void ModelMaterialData::SetMaterial(Material* material)
	{
		m_Material = material;
	}

	List<ModelMaterialData>& ModelImporter::GetMaterials()
	{
		return m_Materials;
	}

	const float& ModelImporter::GetScale()
	{
		return m_Scale;
	}

	void ModelImporter::SetScale(const float& scale)
	{
		m_Scale = scale;
	}

	const bool& ModelImporter::GetGenerateLightmapUV()
	{
		return m_GenerateLightmapUV;
	}

	void ModelImporter::SetGenerateLightmapUV(const bool& generate)
	{
		m_GenerateLightmapUV = generate;
	}

	const bool& ModelImporter::GetGeneratePhysicsShape()
	{
		return m_GeneratePhysicsShape;
	}

	void ModelImporter::SetGeneratePhysicsShape(const bool& generate)
	{
		m_GeneratePhysicsShape = generate;
	}

	void ModelImporter::ImportData()
	{
		Guid guid = GetGuid();
		
		if (AssetDB::HasAssetWithGuidInData(guid))
		{
			// TODO multiple meshes in file
			// TODO rewrite this, it makes allocateid too and add fileid serializing to yamlserializer
			auto objects = AssetDB::LoadAssetObjects(guid, ObjectDB::GetObjectsFromGuid(guid));
			for (auto& pair : objects)
			{
				Object* object = pair.first;
				FileId id = pair.second;
				ObjectDB::AllocateIdToGuid(object, guid, id);
				object->SetState(ObjectState::Default);

				if (object->IsClassType(Mesh::Type))
				{
					Mesh* mesh = static_cast<Mesh*>(object);
					if (m_GeneratePhysicsShape)
					{
						std::ifstream input;
						input.open(GetPhysicsShapePath(id), std::ofstream::binary);
						if (input.is_open())
						{
							PhysicsShapeCache::Load(mesh, input);
							input.close();
						}
					}
					mesh->Apply();
					AddAssetObject(object, id);
					//BB_INFO("Mesh \"" << object->GetName() << "\" imported from cache.");
				}
				else if (object->IsClassType(Entity::Type))
				{
					AddAssetObject(object, id);
					//BB_INFO("Entity \"" << object->GetName() << "\" imported from cache.");
				}
			}
			return;
		}
		else
		{
			// Based on https://www.youtube.com/watch?v=oIKnBVP2Jgg
			fbxsdk::FbxManager* manager = fbxsdk::FbxManager::Create();
			fbxsdk::FbxIOSettings* ios = fbxsdk::FbxIOSettings::Create(manager, IOSROOT);
			manager->SetIOSettings(ios);
			fbxsdk::FbxImporter* importer = fbxsdk::FbxImporter::Create(manager, "");
			if (!importer->Initialize(GetFilePath().c_str(), -1, manager->GetIOSettings()))
			{
				BB_ERROR("Failed to initialize FBX importer: " << importer->GetStatus().GetErrorString());
				return;
			}

			//Matrix rotationMatrix = Matrix::CreateRotationX(ToRadians(-90));

			fbxsdk::FbxScene* scene = fbxsdk::FbxScene::Create(manager, "scene");
			importer->Import(scene);

			fbxsdk::FbxGeometryConverter converter(manager);
			//converter.Triangulate(scene, true); Creates vertexCount == indexCount
			
			int meshCount = 0;
			for (int i = 0; i < scene->GetNodeCount(); ++i)
			{
				if (scene->GetNode(i)->GetMesh() != nullptr)
				{
					++meshCount;
				}
			}

			fbxsdk::FbxNode* rootNode = scene->GetRootNode();
			List<Object*> objects;
			if (meshCount == 1)
			{
				CreateMeshEntity(nullptr, rootNode->GetChild(0), objects);
			}
			else
			{
				CreateMeshEntity(nullptr, rootNode, objects);
			}

			Entity* root = static_cast<Entity*>(ObjectDB::GetObjectFromGuid(GetGuid(), GetMainObject()));

			importer->Destroy();
			manager->Destroy();

			AssetDB::SaveAssetObjectsToCache(objects);
		}
	}

	void ModelImporter::CreateMeshEntity(Transform* parent, fbxsdk::FbxNode* node, List<Object*>& objects)
	{
		if (node->GetLodGroup())
		{
			return;
		}

		Guid guid = GetGuid();
		const auto& importedObjects = ObjectDB::GetObjectsFromGuid(guid);

		String nodeName = node->GetName();
		size_t entityFileId = TO_HASH(String(nodeName).append("_Entity"));
		Entity* entity = nullptr;
		auto it = importedObjects.find(entityFileId);
		if (it != importedObjects.end())
		{
			entity = static_cast<Entity*>(ObjectDB::GetObject(it->second));
			entity->SetState(ObjectState::Default);
		}
		else
		{
			entity = Object::Create<Entity>();
		}
		
		if (nodeName == "RootNode")
		{
			entity->SetName(GetName());
		}
		else
		{
			entity->SetName(nodeName);
		}

		Transform* transform = entity->GetComponent<Transform>();
		if (transform == nullptr)
		{
			transform = Object::Create<Transform>();
			entity->AddComponent(transform);
		}

		ObjectDB::AllocateIdToGuid(entity, guid, entityFileId);
		objects.emplace_back(entity);

		fbxsdk::FbxDouble3 fbxTranslation = node->LclTranslation.Get();
		fbxsdk::FbxDouble3 fbxRotation = node->LclRotation.Get();
		fbxsdk::FbxDouble3 fbxScale = node->LclScaling.Get();

		if (parent != nullptr)
		{
			transform->SetParent(parent);
		}
		else
		{
			SetMainObject(entityFileId);
		}
		transform->SetLocalPosition(Vector3(static_cast<float>(fbxTranslation[0] / fbxScale[0]), static_cast<float>(fbxTranslation[1] / fbxScale[1]), static_cast<float>(fbxTranslation[2] / fbxScale[2])));
		transform->SetLocalEulerRotationHint(Vector3(static_cast<float>(fbxRotation[0]), static_cast<float>(fbxRotation[1]), static_cast<float>(fbxRotation[2])));
		
		if (fbxsdk::FbxMesh* fbxMesh = node->GetMesh())
		{
			if (fbxMesh->RemoveBadPolygons() < 0)
			{
				return;
			}

			if (fbxMesh == nullptr || fbxMesh->RemoveBadPolygons() < 0)
			{
				return;
			}
			
			uint32_t polygonCount = static_cast<uint32_t>(fbxMesh->GetPolygonCount());
			if (polygonCount == 0)
			{
				return;
			}

			size_t meshFileId = TO_HASH(nodeName);
			Mesh* mesh = nullptr;
			it = importedObjects.find(meshFileId);
			if (it != importedObjects.end())
			{
				mesh = static_cast<Mesh*>(ObjectDB::GetObject(it->second));
				mesh->SetState(ObjectState::Default);
			}
			else
			{
				mesh = Mesh::Create();
			}
			ObjectDB::AllocateIdToGuid(mesh, guid, meshFileId);

			MeshRenderer* meshRenderer = entity->GetComponent<MeshRenderer>();
			if (meshRenderer == nullptr)
			{
				meshRenderer = Object::Create<MeshRenderer>();
				entity->AddComponent(meshRenderer);
			}
			// TODO material
			meshRenderer->SetMesh(mesh);

			int fbxControlPointsCount = fbxMesh->GetControlPointsCount();
			fbxsdk::FbxVector4* fbxControlPoints = fbxMesh->GetControlPoints();

			fbxsdk::FbxArray<fbxsdk::FbxVector4> fbxNormals;
			fbxMesh->GetPolygonVertexNormals(fbxNormals);

			fbxsdk::FbxLayerElementArrayTemplate<fbxsdk::FbxVector4>* fbxTangents;
			fbxMesh->GetTangents(&fbxTangents);

			fbxsdk::FbxStringList fbxUvNames;
			fbxMesh->GetUVSetNames(fbxUvNames);
			fbxsdk::FbxArray<fbxsdk::FbxVector2> fbxUvs0;
			fbxsdk::FbxArray<fbxsdk::FbxVector2> fbxUvs1;
			if (fbxUvNames.GetCount() > 0)
			{
				std::string str = fbxUvNames[0];
				fbxMesh->GetPolygonVertexUVs(fbxUvNames[0], fbxUvs0);
			}
			if (fbxUvNames.GetCount() > 1)
			{
				std::string str = fbxUvNames[1];
				fbxMesh->GetPolygonVertexUVs(fbxUvNames[1], fbxUvs1);
			}

			uint32_t verticesCount = static_cast<uint32_t>(fbxMesh->GetPolygonVertexCount());

			static List<Vector3> verticesNormalsTangentsUvs;
			if (verticesCount > static_cast<uint32_t>(verticesNormalsTangentsUvs.size()))
			{
				verticesNormalsTangentsUvs.resize(verticesCount);
			}

			// Vertices
			int* verticesPtr = fbxMesh->mPolygonVertices;
			for (int i = 0, n = fbxMesh->mPolygonVertices.GetCount(); i < n; ++i, ++verticesPtr)
			{
				fbxsdk::FbxVector4 vertex = fbxControlPoints[*verticesPtr];
				// Vector4 point = Vector4::Transform(Vector4(vertex[0], vertex[1], vertex[2], 1.0f), rotationMatrix);
				verticesNormalsTangentsUvs[i] = Vector3(static_cast<float>(vertex[0]), static_cast<float>(vertex[1]), static_cast<float>(vertex[2])) / m_Scale; //  Vector3(vertex[0], vertex[2], -vertex[1])
			}
			mesh->SetVertices(verticesNormalsTangentsUvs.data(), verticesCount);

			// Indices
			uint32_t indicesCount = 0;
			for (uint32_t i = 0; i < polygonCount; ++i)
			{
				// This may break if accidently get 5 or more vertex polygon
				indicesCount += (fbxMesh->mPolygons[i]).mSize == 3 ? 3 : 6;
			}

			static List<uint32_t> indices;
			if (indicesCount > indices.size())
			{
				indices.resize(indicesCount);
			}

			static List<uint32_t> sortedPolygonIndexes;
			sortedPolygonIndexes.reserve(polygonCount);
			sortedPolygonIndexes.clear();

			// Submeshes
			uint32_t materialCount = node->GetMaterialCount();
			List<uint32_t> subMeshPolygonCounts;
			fbxsdk::FbxLayerElementArrayTemplate<int>* materialIndices;
			if (fbxMesh->GetMaterialIndices(&materialIndices))
			{
				uint32_t subMeshStart = 0;
				for (uint32_t i = 0; i < materialCount; ++i)
				{
					uint32_t subMeshSize = 0;
					for (uint32_t j = 0; j < polygonCount; ++j)
					{
						if (materialIndices->GetAt(j) == i)
						{
							sortedPolygonIndexes.emplace_back(j);
							subMeshSize += fbxMesh->mPolygons[j].mSize == 3 ? 3 : 6;
						}
					}
					SubMeshData subMesh = {};
					subMesh.SetIndexStart(subMeshStart);
					subMesh.SetIndexCount(subMeshSize);
					mesh->SetSubMesh(i, subMesh);
					subMeshStart += subMeshSize;
				}
			}
			else
			{
				for (uint32_t i = 0; i < polygonCount; ++i)
				{
					sortedPolygonIndexes.emplace_back(i);
				}
			}

			uint32_t* indicesPtr = indices.data();
			for (uint32_t i = 0; i < polygonCount; ++i)
			{
				fbxsdk::FbxMesh::PolygonDef polygon = fbxMesh->mPolygons[sortedPolygonIndexes[i]];
				if (polygon.mSize == 3)
				{
					*indicesPtr++ = polygon.mIndex;
					*indicesPtr++ = polygon.mIndex + 1;
					*indicesPtr++ = polygon.mIndex + 2;
				}
				if (polygon.mSize == 4)
				{
					*indicesPtr++ = polygon.mIndex;
					*indicesPtr++ = polygon.mIndex + 1;
					*indicesPtr++ = polygon.mIndex + 2;
					*indicesPtr++ = polygon.mIndex;
					*indicesPtr++ = polygon.mIndex + 2;
					*indicesPtr++ = polygon.mIndex + 3;
				}
				else if (polygon.mSize > 4)
				{
					BB_ERROR("Big polygon");
				}
			}
			mesh->SetIndices(indices.data(), indicesCount);

			// Normals
			if (fbxNormals.Size() > 0)
			{
				for (uint32_t i = 0; i < verticesCount; ++i)
				{
					fbxsdk::FbxVector4 fbxNormal = fbxNormals[i];
					//Vector4 direction = Vector4::Transform(Vector4(fbxNormal[0], fbxNormal[1], fbxNormal[2], 0.0f), rotationMatrix);
					verticesNormalsTangentsUvs[i] = Vector3(static_cast<float>(fbxNormal[0]), static_cast<float>(fbxNormal[1]), static_cast<float>(fbxNormal[2])); // Vector3(fbxNormal[0], fbxNormal[2], -fbxNormal[1]);
				}
				mesh->SetNormals(verticesNormalsTangentsUvs.data(), verticesCount);
			}

			// Uvs
			Vector2* uvs = reinterpret_cast<Vector2*>(verticesNormalsTangentsUvs.data());
			if (fbxUvs0.Size() > 0)
			{
				for (uint32_t i = 0; i < verticesCount; ++i)
				{
					fbxsdk::FbxVector2 fbxUv = fbxUvs0[i];
					uvs[i] = Vector2(static_cast<float>(fbxUv[0]), static_cast<float>(fbxUv[1]));
				}
				mesh->SetUVs(0, uvs, verticesCount);
			}

			if (fbxUvs0.Size() > 0 && fbxMesh->GetElementTangentCount() == 0)
			{
				mesh->GenerateTangents();
			}
			if (m_GenerateLightmapUV)
			{
				if (fbxUvs1.Size() > 0)
				{
					Vector3* uvs = reinterpret_cast<Vector3*>(verticesNormalsTangentsUvs.data());
					for (uint32_t i = 0; i < verticesCount; ++i)
					{
						fbxsdk::FbxVector2 fbxUv = fbxUvs1[i];
						uvs[i] = Vector3(static_cast<float>(fbxUv[0]), static_cast<float>(fbxUv[1]), -1.0f);
					}

					uint32_t verticesLeft = verticesCount - 1;
					uint32_t chartIndex = 0;
					std::stack<uint32_t> indexes = {};
					uvs[0].z = static_cast<float>(chartIndex);
					indexes.push(0);

					while (verticesLeft > 0)
					{
						if (indexes.size() == 0)
						{
							for (uint32_t i = 0; i < verticesCount; ++i)
							{
								if (uvs[i].z == -1)
								{
									++chartIndex;
									uvs[i].z = static_cast<float>(chartIndex);
									indexes.push(i);
									--verticesLeft;
									break;
								}
							}
							if (indexes.size() == 0)
							{
								break;
							}
						}

						uint32_t index = indexes.top();
						indexes.pop();
						
						bool foundAdjacent = false;
						for (uint32_t i = 0; i < indicesCount; i += 3)
						{
							uint32_t index1 = indices[i];
							uint32_t index2 = indices[i + 1];
							uint32_t index3 = indices[i + 2];

							if (index == index1 || index == index2 || index == index3)
							{
								if (uvs[index1].z == -1)
								{
									uvs[index1].z = static_cast<float>(chartIndex);
									indexes.push(index1);
									--verticesLeft;
									foundAdjacent = true;
								}

								if (uvs[index2].z == -1)
								{
									uvs[index2].z = static_cast<float>(chartIndex);
									indexes.push(index2);
									--verticesLeft;
									foundAdjacent = true;
								}

								if (uvs[index3].z == -1)
								{
									uvs[index3].z = static_cast<float>(chartIndex);
									indexes.push(index3);
									--verticesLeft;
									foundAdjacent = true;
								}
							}
						}
						if (!foundAdjacent)
						{
							Vector3 uv = uvs[index];
							const float distance = 0.0001f * 0.0001f;
							for (uint32_t i = 0; i < verticesCount; ++i)
							{
								Vector3 uv1 = uvs[i];
								if (uv1.z == -1 && Vector2::DistanceSquared(static_cast<Vector2>(uv), static_cast<Vector2>(uv1)) < distance)
								{
									uvs[i].z = static_cast<float>(chartIndex);
									indexes.push(i);
									--verticesLeft;
								}
							}
						}
					}
					mesh->SetUVs(1, uvs, verticesCount);
				}
				else
				{
					xatlas::Atlas* atlas = xatlas::Create();
					xatlas::MeshDecl decl = {};
					decl.vertexCount = verticesCount;
					decl.vertexPositionData = mesh->GetVertices();
					decl.vertexPositionStride = sizeof(Vector3);
					if (fbxNormals.Size() > 0)
					{
						decl.vertexNormalData = mesh->GetNormals();
						decl.vertexNormalStride = sizeof(Vector3);
					}
					if (fbxUvs0.Size() > 0)
					{
						decl.vertexUvData = mesh->GetUVs(0);
						decl.vertexUvStride = sizeof(Vector2);
					}
					decl.indexCount = indicesCount;
					decl.indexData = mesh->GetIndices();
					decl.indexFormat = xatlas::IndexFormat::UInt32;
					xatlas::AddMeshError error = xatlas::AddMesh(atlas, decl, 1);
					if (error != xatlas::AddMeshError::Success)
					{
						xatlas::Destroy(atlas);
						BB_ERROR("Failed to generate lightmap uv.");
					}
					else
					{
						xatlas::Generate(atlas);
						xatlas::Mesh& atlasMesh = atlas->meshes[0];
						Vector3* uvs = verticesNormalsTangentsUvs.data();
						for (uint32_t i = 0; i < atlasMesh.vertexCount; i++)
						{
							xatlas::Vertex &vertex = atlasMesh.vertexArray[i];
							uvs[vertex.xref] = Vector3(vertex.uv[0] / atlas->width, vertex.uv[1] / atlas->height, static_cast<float>(vertex.chartIndex));
						}
						mesh->SetUVs(1, uvs, verticesCount);
						xatlas::Destroy(atlas);
					}
				}
			}
			if (m_GeneratePhysicsShape)
			{
				std::ofstream output;
				output.open(GetPhysicsShapePath(meshFileId), std::ofstream::binary);
				PhysicsShapeCache::Bake(mesh, output);
				output.close();
			}
			mesh->Apply();

			List<Material*> materials;
			materials.resize(materialCount);
			for (uint32_t i = 0; i < materialCount; ++i)
			{
				fbxsdk::FbxSurfaceMaterial* fbxMaterial = node->GetMaterial(i);
				String name = fbxMaterial->GetName();

				auto index = std::find_if(m_Materials.begin(), m_Materials.end(), [name](ModelMaterialData& d) { return d.GetName() == name; });
				if (index == m_Materials.end())
				{
					ModelMaterialData data = {};
					data.SetName(name);
					m_Materials.emplace_back(data);
				}
				else
				{
					Material* material = index->GetMaterial();
					if (material != nullptr)
					{
						materials[i] = material;
					}
				}
			}
			meshRenderer->SetMaterials(materials);
			
			//BB_INFO("Mesh \"" << nodeName << "\" imported.");
			mesh->SetName(nodeName);
			AddAssetObject(mesh, meshFileId);
			objects.emplace_back(mesh);
		}
		for (int i = 0; i < node->GetChildCount(); ++i)
		{
			fbxsdk::FbxNode* childNode = node->GetChild(i);
			CreateMeshEntity(transform, childNode, objects);
		}
	}

	std::string ModelImporter::GetPhysicsShapePath(const size_t& fileId)
	{
		std::filesystem::path dataPath = Path::GetPhysicsShapeCachePath();
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		dataPath.append(GetGuid().ToString().append(std::to_string(fileId)).append(".shape"));
		return dataPath.string();
	}
}