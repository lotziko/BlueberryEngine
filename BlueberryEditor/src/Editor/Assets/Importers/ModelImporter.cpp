#include "ModelImporter.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Animations\AnimationClip.h"
#include "Blueberry\Physics\PhysicsShapeCache.h"
#include "Blueberry\Tools\FileHelper.h"

#define FBXSDK_SHARED
#include <fbxsdk.h>
#include <xatlas\xatlas.h>
#include <fstream>

#include <directxmesh\DirectXMesh.h>

namespace Blueberry
{
	DATA_DEFINITION(ModelMaterialData)
	{
		DEFINE_FIELD(ModelMaterialData, m_Name, BindingType::String, {})
		DEFINE_FIELD(ModelMaterialData, m_Material, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Material::Type))
	}

	DATA_DEFINITION(ModelAnimationClipData)
	{
		DEFINE_FIELD(ModelAnimationClipData, m_Name, BindingType::String, {})
		DEFINE_FIELD(ModelAnimationClipData, m_ReplaceName, BindingType::String, {})
		DEFINE_FIELD(ModelAnimationClipData, m_FirstFrame, BindingType::Uint, {})
		DEFINE_FIELD(ModelAnimationClipData, m_LastFrame, BindingType::Uint, {})
	}

	OBJECT_DEFINITION(ModelImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(ModelImporter, AssetImporter)
		DEFINE_FIELD(ModelImporter, m_Materials, BindingType::DataList, FieldOptions().SetObjectType(&ModelMaterialData::Type))
		DEFINE_FIELD(ModelImporter, m_AnimationClips, BindingType::DataList, FieldOptions().SetObjectType(&ModelAnimationClipData::Type))
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

	const String& ModelAnimationClipData::GetName()
	{
		return m_Name;
	}

	void ModelAnimationClipData::SetName(const String& name)
	{
		m_Name = name;
	}

	const String& ModelAnimationClipData::GetReplaceName()
	{
		return m_ReplaceName;
	}

	void ModelAnimationClipData::SetReplaceName(const String& replaceName)
	{
		m_ReplaceName = replaceName;
	}

	const uint32_t& ModelAnimationClipData::GetFirstFrame()
	{
		return m_FirstFrame;
	}

	void ModelAnimationClipData::SetFirstFrame(const uint32_t& firstFrame)
	{
		m_FirstFrame = firstFrame;
	}

	const uint32_t& ModelAnimationClipData::GetLastFrame()
	{
		return m_LastFrame;
	}

	void ModelAnimationClipData::SetLastFrame(const uint32_t& lastFrame)
	{
		m_LastFrame = lastFrame;
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

		fbxsdk::FbxScene* scene = fbxsdk::FbxScene::Create(manager, "scene");
		importer->Import(scene);

		FbxAxisSystem engineAxis(FbxAxisSystem::eYAxis, FbxAxisSystem::eParityOdd, FbxAxisSystem::eLeftHanded);
		engineAxis.DeepConvertScene(scene);

		FbxSystemUnit systemUnit = scene->GetGlobalSettings().GetSystemUnit();
		float globalScale = m_Scale * static_cast<float>(systemUnit.GetScaleFactor()) / 100.0f;

		fbxsdk::FbxNode* rootNode = scene->GetRootNode();
		List<Object*> objects;
		Dictionary<fbxsdk::FbxNode*, NodeData> nodeToData;
		size_t prefabInstanceFileId = TO_HASH("PrefabInstance");
		PrefabInstance* instance = GetOrCreateAssetObject<PrefabInstance>(prefabInstanceFileId);
		objects.push_back(instance);

		// Collapse root
		if (rootNode->GetChildCount() == 1)
		{
			rootNode = rootNode->GetChild(0);
		}

		CreateHierarchy(nullptr, rootNode, objects, nodeToData, globalScale);
		for (auto& pair : nodeToData)
		{
			if (pair.first->GetMesh() != nullptr)
			{
				CreateMesh(pair.first, objects, nodeToData, globalScale);
			}
		}
		CreateAnimationClips(scene, objects, globalScale);

		importer->Destroy();
		manager->Destroy();

		instance->OnCreate();
		AssetDB::SaveAssetObjectsToCache(objects);
	}

	Vector4 ConvertVector(FbxVector4 vector)
	{
		return Vector4(static_cast<float>(vector[0]), static_cast<float>(vector[1]), static_cast<float>(vector[2]), static_cast<float>(vector[3]));
	}

	Matrix ConvertMatrix(FbxMatrix matrix)
	{
		return Matrix(ConvertVector(matrix.GetRow(0)), ConvertVector(matrix.GetRow(1)), ConvertVector(matrix.GetRow(2)), ConvertVector(matrix.GetRow(3)));
	}

	Matrix ConvertMatrix(FbxAMatrix matrix)
	{
		return Matrix(ConvertVector(matrix.GetRow(0)), ConvertVector(matrix.GetRow(1)), ConvertVector(matrix.GetRow(2)), ConvertVector(matrix.GetRow(3)));
	}

	void OptimizeMesh(uint32_t& verticesCount, const uint32_t& indicesCount, List<Vector3>& vertices, List<Vector3>& normals, List<Vector2>& uvs0, List<Vector3>& uvs1, List<Vector4>& boneWeights, List<Vector4Uint>& boneIndices, List<uint32_t>& indices)
	{
		// TODO submeshes separate OptimizeVertices
		// https://github.com/microsoft/DirectXMesh/wiki/WeldVertices
		List<uint32_t> pointRep(verticesCount);
		HRESULT hr = DirectX::GenerateAdjacencyAndPointReps(indices.data(), indices.size() / 3, static_cast<DirectX::XMFLOAT3*>(vertices.data()), verticesCount, 0.0f, pointRep.data(), nullptr);
		
		if (FAILED(hr))
		{
			return;
		}

		List<uint32_t> newIndices = indices;
		List<uint32_t> vertexRemap(verticesCount);
		List<uint32_t> reverseVertexRemap(verticesCount);
		DirectX::XMFLOAT3* vertexPtr = vertices.data();
		DirectX::XMFLOAT3* normalPtr = normals.data();
		DirectX::XMFLOAT2* uv0Ptr = uvs0.data();
		DirectX::XMFLOAT3* uv1Ptr = uvs1.data();

		bool hasNormals = normals.size() > 0;
		bool hasUv0 = uvs0.size() > 0;
		bool hasUv1 = uvs1.size() > 0;

		hr = DirectX::WeldVertices(newIndices.data(), newIndices.size() / 3, verticesCount, pointRep.data(), vertexRemap.data(), [&](uint32_t v0, uint32_t v1)
		{
			DirectX::XMVECTOR vA = DirectX::XMLoadFloat3(vertexPtr + v0);
			DirectX::XMVECTOR vB = DirectX::XMLoadFloat3(vertexPtr + v1);

			if (DirectX::XMVector3NotEqual(vA, vB))
			{
				return false;
			}

			if (hasNormals)
			{
				DirectX::XMVECTOR nA = DirectX::XMLoadFloat3(normalPtr + v0);
				DirectX::XMVECTOR nB = DirectX::XMLoadFloat3(normalPtr + v1);

				if (DirectX::XMVector3NotEqual(nA, nB))
				{
					return false;
				}
			}

			if (hasUv0)
			{
				DirectX::XMVECTOR uvA = DirectX::XMLoadFloat2(uv0Ptr + v0);
				DirectX::XMVECTOR uvB = DirectX::XMLoadFloat2(uv0Ptr + v1);

				if (DirectX::XMVector3NotEqual(uvA, uvB))
				{
					return false;
				}
			}

			if (hasUv1)
			{
				DirectX::XMVECTOR uvA = DirectX::XMLoadFloat3(uv1Ptr + v0);
				DirectX::XMVECTOR uvB = DirectX::XMLoadFloat3(uv1Ptr + v1);

				if (DirectX::XMVector3NotEqual(uvA, uvB))
				{
					return false;
				}
			}

			return true;
		});

		if (FAILED(hr))
		{
			return;
		}

		size_t trailingUnused = 0;
		hr = DirectX::OptimizeVertices(newIndices.data(), indicesCount / 3, verticesCount, vertexRemap.data(), &trailingUnused);

		if (FAILED(hr))
		{
			return;
		}

		uint32_t newVerticesCount = static_cast<uint32_t>(verticesCount - trailingUnused);

		for (uint32_t i = 0; i < verticesCount; ++i)
		{
			uint32_t index = vertexRemap[i];
			if (index != UINT32_MAX)
			{
				reverseVertexRemap[index] = i;
			}
		}

		List<Vector3> newVertices(newVerticesCount);
		List<Vector3> newNormals(newVerticesCount);
		for (size_t i = 0; i < newVerticesCount; ++i)
		{
			uint32_t index = vertexRemap[i];
			newVertices[i] = vertices[index];
			newNormals[i] = normals[index];
		}
		vertices = newVertices;
		normals = newNormals;
		if (uvs0.size() > 0)
		{
			List<Vector2> newUvs0(newVerticesCount);
			for (size_t i = 0; i < newVerticesCount; ++i)
			{
				newUvs0[i] = uvs0[vertexRemap[i]];
			}
			uvs0 = newUvs0;
		}
		if (uvs1.size() > 0)
		{
			List<Vector3> newUvs1(newVerticesCount);
			for (size_t i = 0; i < newVerticesCount; ++i)
			{
				newUvs1[i] = uvs1[vertexRemap[i]];
			}
			uvs1 = newUvs1;
		}
		if (boneWeights.size() > 0 && boneIndices.size() > 0)
		{
			List<Vector4> newBoneWeights(newVerticesCount);
			List<Vector4Uint> newBoneIndices(newVerticesCount);
			for (size_t i = 0; i < newVerticesCount; ++i)
			{
				uint32_t index = vertexRemap[i];
				newBoneWeights[i] = boneWeights[index];
				newBoneIndices[i] = boneIndices[index];
			}
			boneWeights = newBoneWeights;
			boneIndices = newBoneIndices;
		}

		for (uint32_t i = 0; i < indicesCount; ++i)
		{
			indices[i] = reverseVertexRemap[newIndices[i]];
		}
		verticesCount = newVerticesCount;
	}

	void ModelImporter::CreateHierarchy(Transform* parent, fbxsdk::FbxNode* node, List<Object*>& objects, Dictionary<fbxsdk::FbxNode*, NodeData>& nodeToData, const float& globalScale)
	{
		String nodeName = node->GetName();
		String entityName = nodeName;
		size_t entityFileId = TO_HASH(String(entityName).append("_Entity"));
		Entity* entity = GetOrCreateAssetObject<Entity>(entityFileId);

		size_t transformFileId = TO_HASH(String(entityName).append("_Transform"));
		Transform* transform = GetOrCreateAssetObject<Transform>(transformFileId);
		if (!entity->HasComponent<Transform>())
		{
			entity->AddComponent(transform);
		}
		if (parent != nullptr)
		{
			transform->SetParent(parent, false);
		}

		bool isRoot = false;
		if (objects.size() <= 1)
		{
			nodeName = "RootNode";
			entityName = GetName();
			SetMainObject(entityFileId);
			objects.push_back(entity);
			isRoot = true;
		}
		entity->SetName(entityName);

		NodeData data = {};
		data.entity = entity;
		data.transform = transform;
		data.nodeName = nodeName;
		data.entityName = entityName;
		nodeToData.insert_or_assign(node, data);

		FbxAMatrix fbxTransform = node->EvaluateLocalTransform();

		fbxsdk::FbxDouble3 fbxTranslation = fbxTransform.GetT();
		fbxsdk::FbxQuaternion fbxRotation = fbxTransform.GetQ();
		fbxsdk::FbxDouble3 fbxScale = fbxTransform.GetS();

		transform->SetLocalPosition(Vector3(static_cast<float>(fbxTranslation[0]) * globalScale, static_cast<float>(fbxTranslation[1]) * globalScale, static_cast<float>(fbxTranslation[2]) * globalScale));
		transform->SetLocalRotation(Quaternion(static_cast<float>(fbxRotation[0]), static_cast<float>(fbxRotation[1]), static_cast<float>(fbxRotation[2]), static_cast<float>(fbxRotation[3])));
		transform->SetLocalScale(Vector3(static_cast<float>(fbxScale[0]), static_cast<float>(fbxScale[1]), static_cast<float>(fbxScale[2])));

		for (int i = 0; i < node->GetChildCount(); ++i)
		{
			fbxsdk::FbxNode* childNode = node->GetChild(i);
			CreateHierarchy(transform, childNode, objects, nodeToData, globalScale);
		}
	}

	void ModelImporter::CreateMesh(fbxsdk::FbxNode* node, List<Object*>& objects, Dictionary<fbxsdk::FbxNode*, NodeData>& nodeToData, const float& globalScale)
	{
		FbxScene* scene = node->GetScene();
		fbxsdk::FbxMesh* fbxMesh = node->GetMesh();
		
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

		NodeData& data = nodeToData[node];
		Entity* entity = data.entity;
		String entityName = data.entityName;
		String nodeName = data.nodeName;
		
		size_t meshFileId = TO_HASH(entityName);
		Mesh* mesh = GetOrCreateAssetObject<Mesh>(meshFileId);

		bool isSkinned = fbxMesh->GetDeformerCount() > 0;
		MeshRenderer* meshRenderer = entity->GetComponent<MeshRenderer>();
		SkinnedMeshRenderer* skinnedMeshRenderer = entity->GetComponent<SkinnedMeshRenderer>();

		if (isSkinned)
		{
			size_t skinnedMeshRendererFileId = TO_HASH(String(nodeName).append("_SkinnedMeshRenderer"));
			skinnedMeshRenderer = GetOrCreateAssetObject<SkinnedMeshRenderer>(skinnedMeshRendererFileId);
			skinnedMeshRenderer->SetMesh(mesh);
			if (!entity->HasComponent<SkinnedMeshRenderer>())
			{
				entity->AddComponent(skinnedMeshRenderer);
			}
			if (meshRenderer != nullptr)
			{
				entity->RemoveComponent(meshRenderer);
			}
		}
		else
		{
			size_t meshRendererFileId = TO_HASH(String(nodeName).append("_MeshRenderer"));
			meshRenderer = GetOrCreateAssetObject<MeshRenderer>(meshRendererFileId);
			meshRenderer->SetMesh(mesh);
			if (!entity->HasComponent<MeshRenderer>())
			{
				entity->AddComponent(meshRenderer);
			}
			if (skinnedMeshRenderer != nullptr)
			{
				entity->RemoveComponent(skinnedMeshRenderer);
			}
		}

		int fbxControlPointsCount = fbxMesh->GetControlPointsCount();
		fbxsdk::FbxVector4* fbxControlPoints = fbxMesh->GetControlPoints();

		fbxsdk::FbxArray<fbxsdk::FbxVector4> fbxNormals;
		fbxMesh->GetPolygonVertexNormals(fbxNormals);

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

		List<Vector3> vertices;
		List<Vector3> normals;
		List<Vector2> uvs0;
		List<Vector3> uvs1;
		List<Vector4> boneWeights;
		List<Vector4Uint> boneIndices;
		List<uint32_t> indices;
		List<SubMeshData> submeshes;

		uint32_t verticesCount = static_cast<uint32_t>(fbxMesh->GetPolygonVertexCount());

		// Vertices
		vertices.resize(verticesCount);
		int* verticesPtr = fbxMesh->mPolygonVertices;
		for (int i = 0, n = fbxMesh->mPolygonVertices.GetCount(); i < n; ++i, ++verticesPtr)
		{
			fbxsdk::FbxVector4 vertex = fbxControlPoints[*verticesPtr];
			vertices[i] = Vector3(static_cast<float>(vertex[0]) * globalScale, static_cast<float>(vertex[1]) * globalScale, static_cast<float>(vertex[2]) * globalScale);
		}

		// Indices
		int indicesCount = 0;
		for (uint32_t i = 0; i < polygonCount; ++i)
		{
			fbxsdk::FbxMesh::PolygonDef polygon = fbxMesh->mPolygons[i];
			if (polygon.mSize == 3)
			{
				indicesCount += 3;
			}
			else if (polygon.mSize == 4)
			{
				indicesCount += 6;
			}
			else
			{
				indicesCount += (polygon.mSize - 2) * 3;
			}
		}
		indices.resize(indicesCount);

		List<uint32_t> sortedPolygonIndexes;

		// Submeshes
		List<uint32_t> materialIndexes;
		fbxsdk::FbxLayerElementArrayTemplate<int>* materialIndices;
		if (fbxMesh->GetMaterialIndices(&materialIndices))
		{
			HashSet<uint32_t> presentMaterialIndexes;
			for (uint32_t i = 0; i < polygonCount; ++i)
			{
				presentMaterialIndexes.insert(materialIndices->GetAt(i));
			}
			if (presentMaterialIndexes.size() > 0)
			{
				uint32_t subMeshStart = 0;
				for (uint32_t materialIndex : presentMaterialIndexes)
				{
					uint32_t subMeshSize = 0;
					for (uint32_t j = 0; j < polygonCount; ++j)
					{
						if (materialIndices->GetAt(j) == materialIndex)
						{
							sortedPolygonIndexes.push_back(j);
							fbxsdk::FbxMesh::PolygonDef polygon = fbxMesh->mPolygons[j];
							if (polygon.mSize == 3)
							{
								subMeshSize += 3;
							}
							else if (polygon.mSize == 4)
							{
								subMeshSize += 6;
							}
							else
							{
								subMeshSize += (polygon.mSize - 2) * 3;
							}
						}
					}
					SubMeshData subMesh = {};
					subMesh.SetIndexStart(subMeshStart);
					subMesh.SetIndexCount(subMeshSize);
					submeshes.push_back(subMesh);
					subMeshStart += subMeshSize;
					materialIndexes.push_back(materialIndex);
				}
			}
		}
		
		if (sortedPolygonIndexes.size() == 0)
		{
			for (uint32_t i = 0; i < polygonCount; ++i)
			{
				sortedPolygonIndexes.push_back(i);
			}
			SubMeshData subMesh = {};
			subMesh.SetIndexStart(0);
			subMesh.SetIndexCount(static_cast<uint32_t>(indices.size()));
			submeshes.push_back(subMesh);
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
			else if (polygon.mSize == 4)
			{
				*indicesPtr++ = polygon.mIndex;
				*indicesPtr++ = polygon.mIndex + 1;
				*indicesPtr++ = polygon.mIndex + 2;
				*indicesPtr++ = polygon.mIndex;
				*indicesPtr++ = polygon.mIndex + 2;
				*indicesPtr++ = polygon.mIndex + 3;
			}
			else
			{
				for (int i = 1; i < polygon.mSize - 1; ++i)
				{
					*indicesPtr++ = polygon.mIndex;
					*indicesPtr++ = polygon.mIndex + i;
					*indicesPtr++ = polygon.mIndex + i + 1;
				}
			}
		}

		// Normals
		if (fbxNormals.Size() > 0)
		{
			normals.resize(verticesCount);
			for (uint32_t i = 0; i < verticesCount; ++i)
			{
				fbxsdk::FbxVector4 fbxNormal = fbxNormals[i];
				normals[i] = Vector3(static_cast<float>(fbxNormal[0]), static_cast<float>(fbxNormal[1]), static_cast<float>(fbxNormal[2])); // Vector3(fbxNormal[0], fbxNormal[2], -fbxNormal[1]);
			}
		}

		// Uvs
		if (fbxUvs0.Size() > 0)
		{
			uvs0.resize(verticesCount);
			for (uint32_t i = 0; i < verticesCount; ++i)
			{
				fbxsdk::FbxVector2 fbxUv = fbxUvs0[i];
				uvs0[i] = Vector2(static_cast<float>(fbxUv[0]), 1.0f - static_cast<float>(fbxUv[1])); // Flip vertically
			}
		}

		if (m_GenerateLightmapUV)
		{
			if (fbxUvs1.Size() > 0)
			{
				const uint32_t triangleCount = indicesCount / 3;
				constexpr float uvEps = 0.0001f;
				constexpr float invUvEps = 1.0f / uvEps;
				uvs1.resize(verticesCount);

				for (uint32_t i = 0; i < verticesCount; ++i)
				{
					fbxsdk::FbxVector2 fbxUv = fbxUvs1[i];
					uvs1[i] = Vector3(static_cast<float>(fbxUv[0]), static_cast<float>(fbxUv[1]), 0);
				}

				Dictionary<Vector2Int, List<uint32_t>> uvToTriangle;
				uvToTriangle.reserve(verticesCount);

				for (uint32_t t = 0; t < triangleCount; ++t)
				{
					for (uint32_t i = 0; i < 3; ++i)
					{
						uint32_t v = indices[t * 3 + i];
						const Vector3& uv = uvs1[v];
						Vector2Int key(static_cast<int32_t>(std::floor(uv.x * invUvEps)), static_cast<int32_t>(std::floor(uv.y * invUvEps)));
						uvToTriangle[key].push_back(t);
					}
				}

				int32_t chartIndex = 0;
				List<uint32_t> stack;
				List<int32_t> triangleCharts(triangleCount, -1);

				for (uint32_t i = 0; i < triangleCount; ++i)
				{
					if (triangleCharts[i] != -1)
					{
						continue;
					}

					triangleCharts[i] = chartIndex;
					stack.push_back(i);

					while (!stack.empty())
					{
						uint32_t cur = stack[stack.size() - 1];
						stack.pop_back();

						for (uint32_t j = 0; j < 3; ++j)
						{
							uint32_t v = indices[cur * 3 + j];
							const Vector3& uv = uvs1[v];
							Vector2Int key(static_cast<int32_t>(std::floor(uv.x * invUvEps)), static_cast<int32_t>(std::floor(uv.y * invUvEps)));

							auto it = uvToTriangle.find(key);
							if (it == uvToTriangle.end())
							{
								continue;
							}

							for (uint32_t n : it->second)
							{
								if (triangleCharts[n] == -1)
								{
									triangleCharts[n] = chartIndex;
									stack.push_back(n);
								}
							}
						}
					}
					++chartIndex;
				}

				for (uint32_t i = 0; i < triangleCount; ++i)
				{
					uint32_t index = i * 3;
					float triangleChart = static_cast<float>(triangleCharts[i]);
					uvs1[indices[index]].z = triangleChart;
					uvs1[indices[index + 1]].z = triangleChart;
					uvs1[indices[index + 2]].z = triangleChart;
				}
			}
			else
			{
				xatlas::Atlas* atlas = xatlas::Create();
				xatlas::MeshDecl decl = {};
				decl.vertexCount = verticesCount;
				decl.vertexPositionData = vertices.data();
				decl.vertexPositionStride = sizeof(Vector3);
				if (normals.size() > 0)
				{
					decl.vertexNormalData = normals.data();
					decl.vertexNormalStride = sizeof(Vector3);
				}
				if (uvs0.size() > 0)
				{
					decl.vertexUvData = uvs0.data();
					decl.vertexUvStride = sizeof(Vector2);
				}
				decl.indexFormat = xatlas::IndexFormat::UInt32;
				bool isValid = true;
				for (auto& submesh : submeshes)
				{
					decl.indexCount = submesh.GetIndexCount();
					decl.indexData = indices.data() + submesh.GetIndexStart();
					xatlas::AddMeshError error = xatlas::AddMesh(atlas, decl, 1);
					if (error != xatlas::AddMeshError::Success)
					{
						xatlas::Destroy(atlas);
						BB_ERROR("Failed to generate lightmap uv.");
						isValid = false;
						break;
					}
				}

				if (isValid)
				{
					List<Vector3> oldVertices = vertices;
					List<Vector3> oldNormals = normals;
					List<Vector2> oldUvs0 = uvs0;

					xatlas::Generate(atlas);
					uvs1.reserve(uvs0.size());
					vertices.clear();
					normals.clear();
					uvs0.clear();
					indices.clear();
					uint32_t vertexOffset = 0;
					uint32_t indexOffset = 0;
					uint32_t chartOffset = 0;
					for (uint32_t i = 0; i < atlas->meshCount; ++i)
					{
						xatlas::Mesh& atlasMesh = atlas->meshes[i];
						auto& submesh = submeshes[i];
						submesh.SetIndexStart(indexOffset);
						submesh.SetIndexCount(atlasMesh.indexCount);
						for (uint32_t j = 0; j < atlasMesh.vertexCount; ++j)
						{
							xatlas::Vertex& vertex = atlasMesh.vertexArray[j];
							uint32_t index = vertex.xref;
							vertices.push_back(oldVertices[index]);
							normals.push_back(oldNormals[index]);
							uvs0.push_back(oldUvs0[index]);
							uvs1.push_back(Vector3(vertex.uv[0] / atlas->width, vertex.uv[1] / atlas->height, static_cast<float>(chartOffset + vertex.chartIndex)));
						}
						for (uint32_t j = 0; j < atlasMesh.indexCount; ++j)
						{
							indices.push_back(vertexOffset + atlasMesh.indexArray[j]);
						}
						vertexOffset += atlasMesh.vertexCount;
						indexOffset += atlasMesh.indexCount;
						chartOffset += atlasMesh.chartCount;
					}
					verticesCount = static_cast<uint32_t>(vertices.size());
					indicesCount = static_cast<uint32_t>(indices.size());
					xatlas::Destroy(atlas);

					Dictionary<uint32_t, uint32_t> chartRemap;
					uint32_t nextChartIndex = 0;

					for (auto& uv : uvs1)
					{
						uint32_t oldIndex = static_cast<uint32_t>(uv.z);
						if (chartRemap.find(oldIndex) == chartRemap.end())
						{
							chartRemap[oldIndex] = nextChartIndex++;
						}
						uv.z = static_cast<float>(chartRemap[oldIndex]);
					}
				}
			}
		}

		if (isSkinned)
		{
			// Skinning
			struct BoneWeight
			{
				float data[4];
			};

			struct BoneIndex
			{
				unsigned int data[4];
			};

			List<BoneWeight> boneWeightControlPoints(fbxControlPointsCount);
			List<BoneIndex> boneIndexControlPoints(fbxControlPointsCount);

			boneWeights.resize(verticesCount);
			boneIndices.resize(verticesCount);

			FbxTime time(0);
			FbxAnimEvaluator* evaluator = scene->GetAnimationEvaluator();
			FbxSkin* skin = static_cast<FbxSkin*>(fbxMesh->GetDeformer(0, fbxsdk::FbxDeformer::EDeformerType::eSkin));
			FbxNode* rootNode = skin->GetCluster(0)->GetLink();
			while (rootNode->GetSkeleton() != nullptr)
			{
				FbxNode* parentNode = rootNode->GetParent();
				if (parentNode == scene->GetRootNode())
				{
					break;
				}
				rootNode = parentNode;
			}

			int clusterCount = skin->GetClusterCount();
			Dictionary<FbxNode*, uint32_t> nodeToIndex;
			{
				for (int i = 0; i < clusterCount; ++i)
				{
					FbxCluster* cluster = skin->GetCluster(i);
					FbxNode* boneNode = cluster->GetLink();
					nodeToIndex.insert_or_assign(boneNode, 0);
				}

				uint32_t offset = 0;
				std::stack<FbxNode*> nodeStack;
				nodeStack.push(rootNode);
				while (nodeStack.size() > 0)
				{
					FbxNode* node = nodeStack.top();
					nodeStack.pop();
					auto it = nodeToIndex.find(node);
					if (it != nodeToIndex.end())
					{
						nodeToIndex.insert_or_assign(node, offset);
						++offset;
					}
					for (int i = 0; i < node->GetChildCount(); ++i)
					{
						nodeStack.push(node->GetChild(i));
					}
				}
			}

			Transform* rootTransform = nodeToData[rootNode].transform;
			List<Transform*> boneTransforms(clusterCount);
			List<Matrix> bindPoses(clusterCount);
			for (int i = 0; i < clusterCount; ++i)
			{
				FbxCluster* cluster = skin->GetCluster(i);
				FbxNode* boneNode = cluster->GetLink();
				Transform* boneTransform = nodeToData[boneNode].transform;
				uint32_t boneIndex = nodeToIndex[boneNode];

				int* indices = cluster->GetControlPointIndices();
				double* weights = cluster->GetControlPointWeights();
				int count = cluster->GetControlPointIndicesCount();

				FbxAMatrix transformLinkMatrix;
				FbxAMatrix transformMatrix;

				cluster->GetTransformMatrix(transformMatrix);
				cluster->GetTransformLinkMatrix(transformLinkMatrix);

				FbxAMatrix bindPoseMatrix = transformLinkMatrix.Inverse() * transformMatrix;
				bindPoseMatrix.SetT(bindPoseMatrix.GetT() * globalScale);
				boneTransforms[boneIndex] = boneTransform;
				bindPoses[boneIndex] = ConvertMatrix(bindPoseMatrix);

				for (int j = 0; j < count; ++j)
				{
					int vertexIndex = indices[j];
					float weight = static_cast<float>(weights[j]);

					auto& indexData = boneIndexControlPoints[vertexIndex].data;
					auto& weightData = boneWeightControlPoints[vertexIndex].data;

					if (weightData[0] == 0.0f)
					{
						weightData[0] = weight;
						indexData[0] = boneIndex;
					}
					else
					{
						for (int k = 0; k < 4; ++k)
						{
							if (weight > weightData[k])
							{
								for (int o = 2; o >= k; --o)
								{
									weightData[o + 1] = weightData[o];
									indexData[o + 1] = indexData[o];
								}
								weightData[k] = weight;
								indexData[k] = boneIndex;
								break;
							}
						}
					}
				}
			}
			mesh->SetBindPoses(bindPoses);

			int* verticesPtr = fbxMesh->mPolygonVertices;
			for (int i = 0, n = fbxMesh->mPolygonVertices.GetCount(); i < n; ++i, ++verticesPtr)
			{
				fbxsdk::FbxVector4 vertex = fbxControlPoints[*verticesPtr];
				Vector4 weight = Vector4(boneWeightControlPoints[*verticesPtr].data);
				float weightSum = weight.x + weight.y + weight.z + weight.w;
				weight.x /= weightSum;
				weight.y /= weightSum;
				weight.z /= weightSum;
				weight.w /= weightSum;
				Vector4Uint indices = Vector4Uint(boneIndexControlPoints[*verticesPtr].data);
				boneWeights[i] = weight;
				boneIndices[i] = indices;
			}

			skinnedMeshRenderer->SetRoot(rootTransform);
			skinnedMeshRenderer->SetBones(boneTransforms);
		}

		OptimizeMesh(verticesCount, indicesCount, vertices, normals, uvs0, uvs1, boneWeights, boneIndices, indices);

		mesh->SetVertices(vertices.data(), verticesCount);
		if (normals.size() > 0)
		{
			mesh->SetNormals(normals.data(), verticesCount);
		}
		if (uvs0.size() > 0)
		{
			mesh->SetUVs(0, uvs0.data(), verticesCount);
		}
		if (uvs1.size() > 0)
		{
			mesh->SetUVs(1, uvs1.data(), verticesCount);
		}
		if (boneWeights.size() > 0 && boneIndices.size() > 0)
		{
			mesh->SetBoneWeights(boneWeights.data(), verticesCount);
			mesh->SetBoneIndices(boneIndices.data(), verticesCount);
		}
		mesh->SetIndices(indices.data(), indicesCount);
		if (uvs0.size() > 0)
		{
			mesh->GenerateTangents();
		}
		for (size_t i = 0; i < submeshes.size(); ++i)
		{
			mesh->SetSubMesh(static_cast<uint32_t>(i), submeshes[i]);
		}
		mesh->Apply();

		if (m_GeneratePhysicsShape)
		{
			PhysicsShapeCache::Clear(mesh);
			// TODO rename to m_GenerateColliders and put colliders on meshes
		}

		List<Material*> materials;
		if (materialIndexes.size() > 0)
		{
			uint32_t materialOffset = 0;
			materials.resize(materialIndexes.size());
			for (uint32_t materialIndex : materialIndexes)
			{
				fbxsdk::FbxSurfaceMaterial* fbxMaterial = node->GetMaterial(materialIndex);
				String name = fbxMaterial->GetName();

				auto index = std::find_if(m_Materials.begin(), m_Materials.end(), [name](ModelMaterialData& d) { return d.GetName() == name; });
				if (index == m_Materials.end())
				{
					ModelMaterialData data = {};
					data.SetName(name);
					m_Materials.push_back(data);
				}
				else
				{
					Material* material = index->GetMaterial();
					if (material != nullptr)
					{
						materials[materialOffset] = material;
					}
				}
				++materialOffset;
			}
		}
		else
		{
			materials.push_back(nullptr);
		}
		if (isSkinned)
		{
			skinnedMeshRenderer->SetMaterials(materials);
		}
		else
		{
			meshRenderer->SetMaterials(materials);
		}

		mesh->SetName(entityName);
		AddAssetObject(mesh, meshFileId);
		objects.push_back(mesh);
	}

	void ModelImporter::CreateAnimationClips(fbxsdk::FbxScene* scene, List<Object*>& objects, const float& globalScale)
	{
		// Animation clips
		int stackCount = scene->GetSrcObjectCount<FbxAnimStack>();
		if (stackCount > 0)
		{
			scene->SetCurrentAnimationStack(nullptr);
			FbxAnimEvaluator* evaluator = scene->GetAnimationEvaluator();
			
			for (int i = 0; i < stackCount; ++i)
			{
				FbxAnimStack* animStack = scene->GetSrcObject<FbxAnimStack>(i);
				String name = animStack->GetName();
				if (name == "Take 001" || name == "Default Take")
				{
					continue;
				}

				// Find nodes
				List<FbxNode*> skeletonNodes;
				{
					FbxAnimLayer* layer = animStack->GetMember<FbxAnimLayer>(0);
					if (layer != nullptr)
					{
						for (int i = 0; i < scene->GetNodeCount(); ++i)
						{
							FbxNode* node = scene->GetNode(i);
							if (node->GetSkeleton() != nullptr)
							{
								if (node->LclTranslation.GetCurveNode(layer) || node->LclRotation.GetCurveNode(layer) || node->LclScaling.GetCurveNode(layer))
								{
									skeletonNodes.push_back(node);
								}
							}
						}
					}
					if (skeletonNodes.size() > 0)
					{
						FbxNode* rootNode = skeletonNodes[0];
						while (rootNode->GetSkeleton() != nullptr)
						{
							FbxNode* parentNode = rootNode->GetParent();
							if (parentNode == scene->GetRootNode())
							{
								break;
							}
							rootNode = parentNode;
						}
					}
				}

				size_t clipFileId = TO_HASH(name);
				AnimationClip* clip = GetOrCreateAssetObject<AnimationClip>(clipFileId);
				clip->ClearAnimationBones();
				scene->SetCurrentAnimationStack(animStack);

				FbxTime start = animStack->LocalStart;
				FbxTime end = animStack->LocalStop;

				double startSec = start.GetSecondDouble();
				double endSec = end.GetSecondDouble();

				FbxTime::EMode timeMode = scene->GetGlobalSettings().GetTimeMode();
				double fps = FbxTime::GetFrameRate(timeMode);
				double dt = 1.0 / fps;

				auto index = std::find_if(m_AnimationClips.begin(), m_AnimationClips.end(), [name](ModelAnimationClipData& d) { return d.GetName() == name; });
				if (index == m_AnimationClips.end())
				{
					ModelAnimationClipData data = {};
					data.SetName(name);
					data.SetReplaceName(name);
					data.SetFirstFrame(0);
					data.SetLastFrame(static_cast<uint32_t>(std::floor(endSec * fps)));
					m_AnimationClips.push_back(data);
				}
				else
				{
					String replaceName = index->GetReplaceName();
					if (name != replaceName)
					{
						name = replaceName;
					}
					double firstFrame = static_cast<double>(index->GetFirstFrame()) * dt;
					if (firstFrame > 0)
					{
						startSec = firstFrame;
					}
					double lastFrame = static_cast<double>(index->GetLastFrame()) * dt;
					if (lastFrame < endSec)
					{
						endSec = lastFrame;
					}
				}

				List<Vector3> positions;
				List<Quaternion> rotations;
				List<Vector3> scales;

				for (size_t j = 0; j < skeletonNodes.size(); ++j)
				{
					FbxTime time;
					FbxNode* skeletonNode = skeletonNodes[j];
					AnimationBoneData boneData = {};
					boneData.SetName(skeletonNode->GetName());

					positions.clear();
					rotations.clear();
					scales.clear();

					for (double t = startSec; t <= endSec; t += dt)
					{
						time.SetSecondDouble(t);
						FbxAMatrix clipMatrix = evaluator->GetNodeLocalTransform(skeletonNode, time);

						FbxVector4 translation = clipMatrix.GetT();
						FbxQuaternion rotation = clipMatrix.GetQ();
						FbxVector4 scale = clipMatrix.GetS();

						positions.push_back(Vector3(static_cast<float>(translation[0]) * globalScale, static_cast<float>(translation[1]) * globalScale, static_cast<float>(translation[2]) * globalScale));
						rotations.push_back(Quaternion(static_cast<float>(rotation[0]), static_cast<float>(rotation[1]), static_cast<float>(rotation[2]), static_cast<float>(rotation[3])));
						scales.push_back(Vector3(static_cast<float>(scale[0]), static_cast<float>(scale[1]), static_cast<float>(scale[2])));
					}

					boneData.SetPositions(positions.data(), positions.size());
					boneData.SetRotations(rotations.data(), rotations.size());
					boneData.SetScales(scales.data(), scales.size());
					clip->AddAnimationBone(boneData);
				}

				clip->SetName(name);
				clip->SetFrameRate(static_cast<float>(fps));
				clip->SetLength(static_cast<float>(endSec - startSec));

				AddAssetObject(clip, clipFileId);
				objects.push_back(clip);
			}
		}
	}
}