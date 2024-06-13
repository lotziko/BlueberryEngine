#include "bbpch.h"
#include "ModelImporter.h"
#include "Editor\Assets\AssetDB.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"

#include "fbxsdk.h"

namespace Blueberry
{
	DATA_DEFINITION(ModelMaterialData)
	OBJECT_DEFINITION(AssetImporter, ModelImporter)

	const std::string& ModelMaterialData::GetName()
	{
		return m_Name;
	}

	void ModelMaterialData::SetName(const std::string& name)
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

	void ModelMaterialData::BindProperties()
	{
		BEGIN_OBJECT_BINDING(ModelMaterialData)
		BIND_FIELD(FieldInfo(TO_STRING(m_Name), &ModelMaterialData::m_Name, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_Material), &ModelMaterialData::m_Material, BindingType::ObjectPtr).SetObjectType(Material::Type))
		END_OBJECT_BINDING()
	}

	const std::vector<DataPtr<ModelMaterialData>>& ModelImporter::GetMaterials()
	{
		return m_Materials;
	}

	void ModelImporter::BindProperties()
	{
		BEGIN_OBJECT_BINDING(ModelImporter)
		BIND_FIELD(FieldInfo(TO_STRING(m_Materials), &ModelImporter::m_Materials, BindingType::DataArray).SetObjectType(ModelMaterialData::Type))
		END_OBJECT_BINDING()
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

			fbxsdk::FbxNode* rootNode = scene->GetRootNode();
			int meshCount = rootNode->GetChildCount();

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
			
			for (int i = 0; i < meshCount; i++)
			{
				fbxsdk::FbxNode* childNode = rootNode->GetChild(i);
				if (childNode->GetLodGroup())
				{
					continue;
				}
				if (fbxsdk::FbxMesh* fbxMesh = childNode->GetMesh())
				{
					if (fbxMesh->RemoveBadPolygons() < 0)
					{
						continue;
					}

					//fbxsdk::FbxGeometryConverter converter(manager);
					//fbxMesh = static_cast<fbxsdk::FbxMesh*>(converter.Triangulate(fbxMesh, true));
					if (fbxMesh == nullptr || fbxMesh->RemoveBadPolygons() < 0)
					{
						continue;
					}

					std::string meshName = childNode->GetName();
					int polygonCount = fbxMesh->GetPolygonCount();
					if (polygonCount <= 0)
					{
						continue;
					}

					Mesh* object = Mesh::Create();
					size_t meshFileId = TO_HASH(meshName);
					ObjectDB::AllocateIdToGuid(object, guid, meshFileId);

					fbxsdk::FbxDouble3 fbxTranslation = childNode->LclTranslation.Get();
					fbxsdk::FbxDouble3 fbxRotation = childNode->LclRotation.Get();
					fbxsdk::FbxDouble3 fbxScale = childNode->LclScaling.Get();

					Entity* entity = Object::Create<Entity>();
					Transform* transform = Object::Create<Transform>();
					if (root != nullptr)
					{
						transform->SetParent(root);
					}
					transform->SetLocalPosition(Vector3(fbxTranslation[0] / fbxScale[0], fbxTranslation[1] / fbxScale[1], fbxTranslation[2] / fbxScale[2]));
					transform->SetLocalRotation(Quaternion::CreateFromYawPitchRoll(ToRadians(fbxRotation[1]), ToRadians(fbxRotation[0] + 90), ToRadians(fbxRotation[2])));

					MeshRenderer* meshRenderer = Object::Create<MeshRenderer>();
					meshRenderer->SetMesh(object);
					// TODO material
					entity->SetName(meshName);
					entity->AddComponent(transform);
					entity->AddComponent(meshRenderer);

					size_t entityFileId = TO_HASH(std::string(meshName).append("_Entity"));
					ObjectDB::AllocateIdToGuid(entity, guid, entityFileId);
					AddImportedObject(entity, entityFileId);
					if (root == nullptr)
					{
						SetMainObject(entityFileId);
					}
					objects.emplace_back(entity);

					int fbxControlPointsCount = fbxMesh->GetControlPointsCount();
					fbxsdk::FbxVector4* fbxControlPoints = fbxMesh->GetControlPoints();

					fbxsdk::FbxArray<fbxsdk::FbxVector4> fbxNormals;
					fbxMesh->GetPolygonVertexNormals(fbxNormals);

					fbxsdk::FbxLayerElementArrayTemplate<fbxsdk::FbxVector4>* fbxTangents;
					fbxMesh->GetTangents(&fbxTangents);

					fbxsdk::FbxStringList fbxUvNames;
					fbxMesh->GetUVSetNames(fbxUvNames);
					fbxsdk::FbxArray<fbxsdk::FbxVector2> fbxUvs;
					if (fbxUvNames.GetCount() > 0)
					{
						fbxMesh->GetPolygonVertexUVs(fbxUvNames[0], fbxUvs);
					}

					int verticesCount = fbxMesh->GetPolygonVertexCount();
					Vector3* verticesNormalsTangentsUvs = new Vector3[verticesCount];

					// Vertices
					int* verticesPtr = fbxMesh->mPolygonVertices;
					for (int i = 0, n = fbxMesh->mPolygonVertices.GetCount(); i < n; ++i, ++verticesPtr)
					{
						fbxsdk::FbxVector4 vertex = fbxControlPoints[*verticesPtr];
						// Vector4 point = Vector4::Transform(Vector4(vertex[0], vertex[1], vertex[2], 1.0f), rotationMatrix);
						verticesNormalsTangentsUvs[i] = Vector3(vertex[0], vertex[2], -vertex[1]);
					}
					object->SetVertices(verticesNormalsTangentsUvs, verticesCount);

					// Indices
					int indicesCount = 0;
					fbxsdk::FbxMesh::PolygonDef* polygonPtr = fbxMesh->mPolygons;
					for (int i = 0, n = fbxMesh->GetPolygonCount(); i < n; ++i, ++polygonPtr)
					{
						indicesCount += polygonPtr->mSize == 3 ? 3 : 6;
					}
					polygonPtr = fbxMesh->mPolygons;
					UINT* indices = new UINT[indicesCount];
					UINT* indicesPtr = indices;
					for (int i = 0, n = fbxMesh->GetPolygonCount(); i < n; ++i, ++polygonPtr)
					{
						fbxsdk::FbxMesh::PolygonDef polygon = *polygonPtr;
						if (polygon.mSize == 3)
						{
							*indicesPtr++ = polygon.mIndex;
							*indicesPtr++ = polygon.mIndex + 1;
							*indicesPtr++ = polygon.mIndex + 2;
						}
						else
						{
							*indicesPtr++ = polygon.mIndex;
							*indicesPtr++ = polygon.mIndex + 1;
							*indicesPtr++ = polygon.mIndex + 2;
							*indicesPtr++ = polygon.mIndex;
							*indicesPtr++ = polygon.mIndex + 2;
							*indicesPtr++ = polygon.mIndex + 3;
						}
					}
					object->SetIndices(indices, indicesCount);
					delete[] indices;

					// Normals
					if (fbxNormals.Size() > 0)
					{
						for (int i = 0; i < verticesCount; ++i)
						{
							fbxsdk::FbxVector4 fbxNormal = fbxNormals[i];
							//Vector4 direction = Vector4::Transform(Vector4(fbxNormal[0], fbxNormal[1], fbxNormal[2], 0.0f), rotationMatrix);
							verticesNormalsTangentsUvs[i] = Vector3(fbxNormal[0], fbxNormal[2], -fbxNormal[1]);
						}
						object->SetNormals(verticesNormalsTangentsUvs, verticesCount);
					}

					// Uvs
					Vector2* uvs = (Vector2*)verticesNormalsTangentsUvs;
					if (fbxUvs.Size() > 0)
					{
						for (int i = 0; i < verticesCount; ++i)
						{
							fbxsdk::FbxVector2 fbxUv = fbxUvs[i];
							uvs[i] = Vector2(fbxUv[0], fbxUv[1]);
						}
						object->SetUVs(0, uvs, verticesCount);
					}
					delete[] verticesNormalsTangentsUvs;

					if (fbxUvs.Size() > 0)
					{
						object->GenerateTangents();
					}
					object->Apply();

					for (int i = 0; i < childNode->GetMaterialCount(); i++)
					{
						fbxsdk::FbxSurfaceMaterial* fbxMaterial = childNode->GetMaterial(i);
						std::string name = fbxMaterial->GetName();

						auto index = std::find_if(m_Materials.begin(), m_Materials.end(), [name](DataPtr<ModelMaterialData> const& d) { return d.Get()->GetName() == name; });
						if (index == m_Materials.end())
						{
							ModelMaterialData* data = new ModelMaterialData();
							data->SetName(name);
							m_Materials.emplace_back(data);
						}
						else
						{
							Material* material = index->Get()->GetMaterial();
							if (material != nullptr)
							{
								meshRenderer->SetMaterial(material);
							}
						}
					}

					BB_INFO("Mesh \"" << meshName << "\" imported.");
					object->SetName(meshName);
					AddImportedObject(object, meshFileId);
					objects.emplace_back(object);
				}
			}

			importer->Destroy();
			manager->Destroy();

			AssetDB::SaveAssetObjectsToCache(objects);
		}
	}

	std::string ModelImporter::GetIconPath()
	{
		return "assets/icons/FbxIcon.png";
	}
}