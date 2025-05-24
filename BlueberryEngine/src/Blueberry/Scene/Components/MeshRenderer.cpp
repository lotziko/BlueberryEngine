#include "Blueberry\Scene\Components\MeshRenderer.h"

#include "..\Scene.h"
#include "Blueberry\Core\ClassDB.h"
#include "..\..\Graphics\RendererTree.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Entity.h"

#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	OBJECT_DEFINITION(MeshRenderer, Renderer)
	{
		DEFINE_BASE_FIELDS(MeshRenderer, Renderer)
		DEFINE_FIELD(MeshRenderer, m_Mesh, BindingType::ObjectPtr, FieldOptions().SetObjectType(Mesh::Type))
		DEFINE_FIELD(MeshRenderer, m_Materials, BindingType::ObjectPtrArray, FieldOptions().SetObjectType(Material::Type))
	}

	void MeshRenderer::OnEnable()
	{
		AddToSceneComponents(MeshRenderer::Type);
		// TODO handle prefabs
		Scene* scene = GetScene();
		if (scene != nullptr)
		{
			m_Bounds = GetBounds();
			m_PreviousBounds = m_Bounds;
			scene->GetRendererTree().Add(m_ObjectId, m_Bounds);
		}
	}

	void MeshRenderer::OnDisable()
	{
		RemoveFromSceneComponents(MeshRenderer::Type);
		Scene* scene = GetScene();
		if (scene != nullptr)
		{
			scene->GetRendererTree().Remove(m_ObjectId, m_Bounds);
		}
	}

	void MeshRenderer::OnPreCull()
	{
		UpdateBounds();
		if (m_CullingDirty)
		{
			GetScene()->GetRendererTree().Update(m_ObjectId, m_PreviousBounds, m_Bounds);
			m_CullingDirty = false;
		}
	}

	Mesh* MeshRenderer::GetMesh()
	{
		return m_Mesh.Get();
	}

	void MeshRenderer::SetMesh(Mesh* mesh)
	{
		m_Mesh = mesh;
		m_RecalculationFrame = 0;
	}

	Material* MeshRenderer::GetMaterial(const uint32_t& index)
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

	void MeshRenderer::SetMaterials(const List<Material*> materials)
	{
		m_Materials.clear();
		for (Material* material : materials)
		{
			m_Materials.emplace_back(material);
		}
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

	void MeshRenderer::UpdateBounds()
	{
		Transform* transform = GetTransform();
		size_t transformRecalculationFrame = transform->GetRecalculationFrame();
		if (m_RecalculationFrame < transformRecalculationFrame)
		{
			AABB bounds = m_Mesh->GetBounds();
			Matrix matrix = transform->GetLocalToWorldMatrix();

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
			m_RecalculationFrame = transformRecalculationFrame;
		}
	}
}
