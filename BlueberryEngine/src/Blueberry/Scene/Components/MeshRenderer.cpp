#include "Blueberry\Scene\Components\MeshRenderer.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxBottomLevelAccelerationStructure.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	OBJECT_DEFINITION(MeshRenderer, Renderer)
	{
		DEFINE_BASE_FIELDS(MeshRenderer, Renderer)
		DEFINE_FIELD(MeshRenderer, m_Mesh, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Mesh::Type).SetUpdateCallback(MethodBind::Create(&MeshRenderer::InvalidateBounds)))
		DEFINE_FIELD(MeshRenderer, m_Materials, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&Material::Type))
		DEFINE_FIELD(MeshRenderer, m_IsBakeable, BindingType::Bool, FieldOptions())
		DEFINE_FIELD(MeshRenderer, m_LightmapChartOffset, BindingType::Uint, FieldOptions().SetVisibility(VisibilityType::Hidden).SetSerializationFlags(SerializationFlags::RuntimeOnly))
		DEFINE_ITERATOR(MeshRenderer)
		DEFINE_EXECUTE_ALWAYS()
	}

	void MeshRenderer::OnEnable()
	{
		Scene* scene = GetScene();
		if (scene != nullptr)
		{
			m_BoundsDirty = true;
			m_Bounds = GetBounds();
			scene->GetRendererTree().Add(this, m_Bounds);
			GetTransform()->AddDependency(this);
		}
	}

	void MeshRenderer::OnDisable()
	{
		Scene* scene = GetScene();
		if (scene != nullptr)
		{
			scene->GetRendererTree().Remove(this);
			GetTransform()->RemoveDependency(this);
		}
	}

	void MeshRenderer::OnTransformInvalidate()
	{
		InvalidateBounds();
	}

	void MeshRenderer::OnPreCull()
	{
		if (m_IsActive)
		{
			UpdateBounds();
			GetScene()->GetRendererTree().Update(this, m_Bounds);
		}
	}

	Mesh* MeshRenderer::GetMesh()
	{
		return m_Mesh.Get();
	}

	void MeshRenderer::SetMesh(Mesh* mesh)
	{
		m_Mesh = mesh;
		InvalidateBounds();
	}

	Material* MeshRenderer::GetMaterial(uint32_t index) const
	{
		if (index >= m_Materials.size())
		{
			return nullptr;
		}
		return m_Materials[index].Get();
	}

	void MeshRenderer::SetMaterial(Material* material)
	{
		if (m_Materials.size() == 0)
		{
			m_Materials.resize(1);
		}
		m_Materials[0] = material;
	}

	const List<ObjectPtr<Material>>& MeshRenderer::GetMaterials() const
	{
		return m_Materials;
	}

	void MeshRenderer::SetMaterials(const List<Material*> materials)
	{
		m_Materials.clear();
		for (Material* material : materials)
		{
			m_Materials.push_back(material);
		}
	}

	uint32_t MeshRenderer::GetMaterialCount() const
	{
		return static_cast<uint32_t>(m_Materials.size());
	}

	const AABB& MeshRenderer::GetBounds()
	{
		if (!m_Mesh.IsValid())
		{
			return m_Bounds;
		}
		
		UpdateBounds();
		return m_Bounds;
	}

	const Matrix& MeshRenderer::GetLocalToWorldMatrix()
	{
		return GetTransform()->GetLocalToWorldMatrix();
	}

	const bool& MeshRenderer::IsBakeable()
	{
		return m_IsBakeable;
	}

	uint32_t MeshRenderer::GetLightmapChartOffset() const
	{
		return m_LightmapChartOffset;
	}

	void MeshRenderer::SetLightmapChartOffset(uint32_t offset)
	{
		m_LightmapChartOffset = offset;
	}

	GfxBottomLevelAccelerationStructure* MeshRenderer::GetAccelerationStructure()
	{
		if (m_Mesh.IsValid())
		{
			uint32_t meshUpdateCount = m_Mesh->GetUpdateCount();
			if (m_MeshUpdateCount != meshUpdateCount)
			{
				m_AccelerationStructure = nullptr;
			}
			if (m_AccelerationStructure == nullptr)
			{
				bool opaqueMask[16] = {};
				for (size_t i = 0; i < m_Materials.size(); ++i)
				{
					opaqueMask[i] = m_Materials[i]->IsOpaque();
				}
				m_AccelerationStructure = GfxBottomLevelAccelerationStructure::Get(m_Mesh.Get(), opaqueMask);
				m_MeshUpdateCount = meshUpdateCount;
			}
		}
		return m_AccelerationStructure;
	}

	void MeshRenderer::UpdateBounds()
	{
		if (m_Mesh.IsValid())
		{
			if (m_BoundsDirty)
			{
				AABB bounds = m_Mesh->GetBounds();
				Matrix matrix = GetTransform()->GetLocalToWorldMatrix();
				bounds.Transform(m_Bounds, matrix);
				m_BoundsDirty = false;
			}
		}
	}

	void MeshRenderer::InvalidateBounds()
	{
		m_BoundsDirty = true;
		Scene* scene = GetScene();
		if (scene != nullptr)
		{
			scene->MarkCullableDirty(m_ObjectId, this);
		}
	}
}
