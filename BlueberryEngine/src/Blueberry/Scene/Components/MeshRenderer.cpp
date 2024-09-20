#include "bbpch.h"
#include "MeshRenderer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Entity.h"

#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Renderer, MeshRenderer)

	void MeshRenderer::OnEnable()
	{
		AddToSceneComponents(MeshRenderer::Type);
	}

	void MeshRenderer::OnDisable()
	{
		RemoveFromSceneComponents(MeshRenderer::Type);
	}

	Mesh* MeshRenderer::GetMesh()
	{
		return m_Mesh.Get();
	}

	void MeshRenderer::SetMesh(Mesh* mesh)
	{
		m_Mesh = mesh;
	}

	Material* MeshRenderer::GetMaterial(const UINT& index)
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

	void MeshRenderer::SetMaterials(const std::vector<Material*> materials)
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

		Transform* transform = GetTransform();
		size_t transformRecalculationFrame = transform->GetRecalculationFrame();
		if (m_RecalculationFrame <= transformRecalculationFrame)
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
			m_Bounds = bounds;
			m_RecalculationFrame = transformRecalculationFrame;
		}
		return m_Bounds;
	}

	void MeshRenderer::BindProperties()
	{
		BEGIN_OBJECT_BINDING(MeshRenderer)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &MeshRenderer::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Mesh), &MeshRenderer::m_Mesh, BindingType::ObjectPtr).SetObjectType(Mesh::Type))
		//BIND_FIELD(FieldInfo(TO_STRING(m_Material), &MeshRenderer::m_Material, BindingType::ObjectPtr).SetObjectType(Material::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Materials), &MeshRenderer::m_Materials, BindingType::ObjectPtrArray).SetObjectType(Material::Type))
		END_OBJECT_BINDING()
	}
}
