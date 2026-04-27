#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Time.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Graphics\GfxBuffer.h"

namespace Blueberry
{
	OBJECT_DEFINITION(SkinnedMeshRenderer, Renderer)
	{
		DEFINE_BASE_FIELDS(SkinnedMeshRenderer, Renderer)
		DEFINE_FIELD(SkinnedMeshRenderer, m_Mesh, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Mesh::Type).SetUpdateCallback(MethodBind::Create(&SkinnedMeshRenderer::InvalidateBounds)))
		DEFINE_FIELD(SkinnedMeshRenderer, m_Materials, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&Material::Type))
		DEFINE_FIELD(SkinnedMeshRenderer, m_Root, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Transform::Type))
		DEFINE_FIELD(SkinnedMeshRenderer, m_Bones, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&Transform::Type).SetVisibility(VisibilityType::NonExposed))
		DEFINE_ITERATOR(SkinnedMeshRenderer)
		DEFINE_EXECUTE_ALWAYS()
	}

	void SkinnedMeshRenderer::OnEnable()
	{
		Scene* scene = GetScene();
		if (scene != nullptr)
		{
			m_Bounds = GetBounds();
			m_PreviousBounds = m_Bounds;
			scene->GetRendererTree().Add(m_ObjectId, m_Bounds);
		}
		UpdateBoneDatas();
	}

	void SkinnedMeshRenderer::OnDisable()
	{
		Scene* scene = GetScene();
		if (scene != nullptr)
		{
			scene->GetRendererTree().Remove(m_ObjectId, m_Bounds);
		}

		if (m_SkinningVertexBuffer != nullptr)
		{
			delete m_SkinningVertexBuffer;
			m_SkinningVertexBuffer = nullptr;
		}
	}

	void SkinnedMeshRenderer::OnPreCull()
	{
		UpdateBounds();
		if (m_CullingDirty)
		{
			GetScene()->GetRendererTree().Update(m_ObjectId, m_PreviousBounds, m_Bounds);
			m_CullingDirty = false;
		}
	}

	Mesh* SkinnedMeshRenderer::GetMesh()
	{
		return m_Mesh.Get();
	}

	void SkinnedMeshRenderer::SetMesh(Mesh* mesh)
	{
		m_Mesh = mesh;
		InvalidateBounds();
	}

	Material* SkinnedMeshRenderer::GetMaterial(const uint32_t& index)
	{
		if (index >= m_Materials.size())
		{
			return nullptr;
		}
		return m_Materials[index].Get();
	}

	void SkinnedMeshRenderer::SetMaterial(Material* material)
	{
		if (m_Materials.size() == 0)
		{
			m_Materials.resize(1);
		}
		m_Materials[0] = material;
	}

	void SkinnedMeshRenderer::SetMaterials(const List<Material*> materials)
	{
		m_Materials.clear();
		for (Material* material : materials)
		{
			m_Materials.push_back(material);
		}
	}

	uint32_t SkinnedMeshRenderer::GetMaterialCount()
	{
		return static_cast<uint32_t>(m_Materials.size());
	}

	const bool SkinnedMeshRenderer::HasRoot()
	{
		return m_Root.IsValid() && m_Root->GetParent() != nullptr;
	}

	Transform* SkinnedMeshRenderer::GetRoot()
	{
		return m_Root.Get();
	}

	void SkinnedMeshRenderer::SetRoot(Transform* root)
	{
		m_Root = root;
	}

	void SkinnedMeshRenderer::SetBones(const List<Transform*>& bones)
	{
		m_Bones.clear();
		for (size_t i = 0; i < bones.size(); ++i)
		{
			m_Bones.push_back(bones[i]);
		}
		UpdateBoneDatas();
	}

	const AABB& SkinnedMeshRenderer::GetBounds()
	{
		if (!m_Mesh.IsValid())
		{
			return m_Bounds;
		}

		UpdateBounds();
		return m_Bounds;
	}

	const Matrix& SkinnedMeshRenderer::GetLocalToWorldMatrix()
	{
		return m_Root->GetParent()->GetLocalToWorldMatrix();
	}
	
	const List<Matrix>& SkinnedMeshRenderer::GetWorldMatrices()
	{
		return m_WorldMatrices;
	}

	const List<Matrix>& SkinnedMeshRenderer::GetSkinningMatrices()
	{
		return m_SkinningMatrices;
	}

	bool SkinnedMeshRenderer::CalculateSkinning()
	{
		size_t frameCount = Time::GetFrameCount();
		if (m_LastVisibleFrame < frameCount)
		{
			size_t boneCount = m_Bones.size();
			if (boneCount > 0)
			{
				if (m_SkinningMatrices.size() != boneCount)
				{
					m_LocalMatrices.resize(boneCount);
					m_WorldMatrices.resize(boneCount);
					m_SkinningMatrices.resize(boneCount);
				}

				for (size_t i = 0; i < boneCount; ++i)
				{
					Transform* transform = m_BoneDatas[i].transform.Get();
					m_LocalMatrices[i] = transform->GetLocalMatrix();
				}

				for (size_t i = 0; i < boneCount; ++i)
				{
					int32_t parentIndex = m_BoneDatas[i].parentIndex;
					if (parentIndex >= 0)
					{
						m_WorldMatrices[i] = m_LocalMatrices[i] * m_WorldMatrices[parentIndex];
					}
					else
					{
						if (parentIndex == -1)
						{
							m_WorldMatrices[i] = m_LocalMatrices[i] * m_Root.Get()->GetLocalMatrix();
						}
						else
						{
							m_WorldMatrices[i] = m_LocalMatrices[i];
						}
					}
				}

				for (size_t i = 0; i < boneCount; ++i)
				{
					m_SkinningMatrices[i] = m_Mesh->GetBindPose(i) * m_WorldMatrices[i];
				}
				m_LastVisibleFrame = frameCount;
				return true;
			}
		}
		return false;
	}

	bool SkinnedMeshRenderer::IsSkinningBufferValid()
	{
		if (m_SkinningVertexBuffer == nullptr)
		{
			return false;
		}
		if (m_SkinningMeshId != m_Mesh->GetObjectId())
		{
			return false;
		}
		if (m_SkinningMeshUpdateCount != m_Mesh->GetUpdateCount())
		{
			return false;
		}
		return true;
	}

	void SkinnedMeshRenderer::UpdateBounds()
	{
		if (m_Mesh.IsValid())
		{
			Transform* transform = GetRoot();
			size_t transformUpdateCount = transform->GetUpdateCount();
			if (m_UpdateCount != transformUpdateCount && m_Bones.size() > 0)
			{
				AABB meshBounds = m_Mesh->GetBounds();
				AABB bounds(Vector3::Zero, Vector3(meshBounds.Extents.x * 2, meshBounds.Extents.y * 2, meshBounds.Extents.z * 2));
				Matrix matrix = m_Bones[0]->GetLocalToWorldMatrix();

				Vector3 corners[8];
				bounds.GetCorners(corners);

				for (int i = 0; i < 8; i++)
				{
					Vector3 corner = corners[i];
					Vector3::Transform(corner, matrix, corners[i]);
				}

				AABB::CreateFromPoints(bounds, 8, corners, sizeof(Vector3));
				if (!m_CullingDirty)
				{
					m_PreviousBounds = m_Bounds;
					m_CullingDirty = true;
				}
				m_Bounds = bounds;
				m_UpdateCount = transformUpdateCount;
			}
		}
	}

	void SkinnedMeshRenderer::InvalidateBounds()
	{
		m_UpdateCount = 0;
	}

	void SkinnedMeshRenderer::UpdateBoneDatas()
	{
		m_BoneDatas.clear();
		Transform* rootBone = m_Root.Get();
		for (size_t i = 0; i < m_Bones.size(); ++i)
		{
			Transform* bone = m_Bones[i].Get();
			if (bone != nullptr)
			{
				int32_t index = -1;
				if (bone == rootBone)
				{
					index = -2;
					m_BoneDatas.push_back({ bone, index });
					continue;
				}
				Transform* boneParent = bone->GetParent();
				for (size_t j = 0; j < i; ++j)
				{
					if (m_Bones[j].Get() == boneParent)
					{
						index = static_cast<int32_t>(j);
						break;
					}
				}
				m_BoneDatas.push_back({ bone, index });
			}
		}
	}
}
