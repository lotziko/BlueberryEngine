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

	const float& ModelImporter::GetScale()
	{
		return m_Scale;
	}

	void ModelImporter::SetScale(const float& scale)
	{
		m_Scale = scale;
	}

	void ModelImporter::BindProperties()
	{
		BEGIN_OBJECT_BINDING(ModelImporter)
		BIND_FIELD(FieldInfo(TO_STRING(m_Materials), &ModelImporter::m_Materials, BindingType::DataArray).SetObjectType(ModelMaterialData::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Scale), &ModelImporter::m_Scale, BindingType::Float))
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
				BB_INFO(id);
				ObjectDB::AllocateIdToGuid(object, guid, id);
				object->SetState(ObjectState::Default);

				if (object->IsClassType(Mesh::Type))
				{
					(static_cast<Mesh*>(object))->Apply();
					AddImportedObject(object, id);
					//BB_INFO("Mesh \"" << object->GetName() << "\" imported from cache.");
				} 
				else if (object->IsClassType(Entity::Type))
				{
					AddImportedObject(object, id);
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
			
			int meshCount = 0;
			for (int i = 0; i < scene->GetNodeCount(); ++i)
			{
				if (scene->GetNode(i)->GetMesh() != nullptr)
				{
					++meshCount;
				}
			}

			fbxsdk::FbxNode* rootNode = scene->GetRootNode();
			std::vector<Object*> objects;
			if (meshCount == 1)
			{
				CreateMeshEntity(nullptr, rootNode->GetChild(0), objects);
			}
			else
			{
				CreateMeshEntity(nullptr, rootNode, objects);
			}

			Entity* root = (Entity*)ObjectDB::GetObjectFromGuid(GetGuid(), GetMainObject());

			importer->Destroy();
			manager->Destroy();

			AssetDB::SaveAssetObjectsToCache(objects);
		}
	}

	std::string ModelImporter::GetIconPath()
	{
		return "assets/icons/FbxIcon.png";
	}

	void ModelImporter::CreateMeshEntity(Transform* parent, fbxsdk::FbxNode* node, std::vector<Object*>& objects)
	{
		if (node->GetLodGroup())
		{
			return;
		}

		Guid guid = GetGuid();
		auto& importedObjects = GetImportedObjects();

		std::string nodeName = node->GetName();
		size_t entityFileId = TO_HASH(std::string(nodeName).append("_Entity"));
		Entity* entity = nullptr;
		auto it = importedObjects.find(entityFileId);
		if (it != importedObjects.end())
		{
			entity = (Entity*)ObjectDB::GetObject(it->second);
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
		AddImportedObject(entity, entityFileId);
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
		transform->SetLocalPosition(Vector3(fbxTranslation[0] / fbxScale[0], fbxTranslation[1] / fbxScale[1], fbxTranslation[2] / fbxScale[2]));
		transform->SetLocalEulerRotationHint(Vector3(fbxRotation[0], fbxRotation[1], fbxRotation[2]));
		
		if (fbxsdk::FbxMesh* fbxMesh = node->GetMesh())
		{
			if (fbxMesh->RemoveBadPolygons() < 0)
			{
				return;
			}

			//fbxsdk::FbxGeometryConverter converter(manager);
			//fbxMesh = static_cast<fbxsdk::FbxMesh*>(converter.Triangulate(fbxMesh, true));
			if (fbxMesh == nullptr || fbxMesh->RemoveBadPolygons() < 0)
			{
				return;
			}
			
			int polygonCount = fbxMesh->GetPolygonCount();
			if (polygonCount <= 0)
			{
				return;
			}

			size_t meshFileId = TO_HASH(nodeName);
			Mesh* mesh = nullptr;
			it = importedObjects.find(meshFileId);
			if (it != importedObjects.end())
			{
				mesh = (Mesh*)ObjectDB::GetObject(it->second);
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
			fbxsdk::FbxArray<fbxsdk::FbxVector2> fbxUvs;
			if (fbxUvNames.GetCount() > 0)
			{
				fbxMesh->GetPolygonVertexUVs(fbxUvNames[0], fbxUvs);
			}

			int verticesCount = fbxMesh->GetPolygonVertexCount();

			static std::vector<Vector3> verticesNormalsTangentsUvs;
			verticesNormalsTangentsUvs.reserve(verticesCount);

			// Vertices
			int* verticesPtr = fbxMesh->mPolygonVertices;
			for (int i = 0, n = fbxMesh->mPolygonVertices.GetCount(); i < n; ++i, ++verticesPtr)
			{
				fbxsdk::FbxVector4 vertex = fbxControlPoints[*verticesPtr];
				// Vector4 point = Vector4::Transform(Vector4(vertex[0], vertex[1], vertex[2], 1.0f), rotationMatrix);
				verticesNormalsTangentsUvs[i] = Vector3(vertex[0], vertex[1], vertex[2]) / m_Scale; //  Vector3(vertex[0], vertex[2], -vertex[1])
			}
			mesh->SetVertices(verticesNormalsTangentsUvs.data(), verticesCount);

			// Indices
			int indicesCount = 0;
			fbxsdk::FbxMesh::PolygonDef* polygonPtr = fbxMesh->mPolygons;
			for (int i = 0, n = fbxMesh->GetPolygonCount(); i < n; ++i, ++polygonPtr)
			{
				indicesCount += polygonPtr->mSize == 3 ? 3 : 6;
			}
			polygonPtr = fbxMesh->mPolygons;

			static std::vector<UINT> indices;
			indices.reserve(indicesCount);

			UINT* indicesPtr = indices.data();
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
			mesh->SetIndices(indices.data(), indicesCount);

			// Normals
			if (fbxNormals.Size() > 0)
			{
				for (int i = 0; i < verticesCount; ++i)
				{
					fbxsdk::FbxVector4 fbxNormal = fbxNormals[i];
					//Vector4 direction = Vector4::Transform(Vector4(fbxNormal[0], fbxNormal[1], fbxNormal[2], 0.0f), rotationMatrix);
					verticesNormalsTangentsUvs[i] = Vector3(fbxNormal[0], fbxNormal[1], fbxNormal[2]); // Vector3(fbxNormal[0], fbxNormal[2], -fbxNormal[1]);
				}
				mesh->SetNormals(verticesNormalsTangentsUvs.data(), verticesCount);
			}

			// Uvs
			Vector2* uvs = (Vector2*)verticesNormalsTangentsUvs.data();
			if (fbxUvs.Size() > 0)
			{
				for (int i = 0; i < verticesCount; ++i)
				{
					fbxsdk::FbxVector2 fbxUv = fbxUvs[i];
					uvs[i] = Vector2(fbxUv[0], fbxUv[1]);
				}
				mesh->SetUVs(0, uvs, verticesCount);
			}

			if (fbxUvs.Size() > 0 && fbxMesh->GetElementTangentCount() == 0)
			{
				mesh->GenerateTangents();
			}
			mesh->Apply();

			for (int i = 0; i < node->GetMaterialCount(); ++i)
			{
				fbxsdk::FbxSurfaceMaterial* fbxMaterial = node->GetMaterial(i);
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

			//BB_INFO("Mesh \"" << nodeName << "\" imported.");
			mesh->SetName(nodeName);
			AddImportedObject(mesh, meshFileId);
			objects.emplace_back(mesh);
		}
		for (int i = 0; i < node->GetChildCount(); ++i)
		{
			fbxsdk::FbxNode* childNode = node->GetChild(i);
			CreateMeshEntity(transform, childNode, objects);
		}
	}
}