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

	Mesh* MeshRenderer::GetMesh()
	{
		return m_Mesh.Get();
	}

	void MeshRenderer::SetMesh(Mesh* mesh)
	{
		m_Mesh = mesh;
	}

	Material* MeshRenderer::GetMaterial()
	{
		return m_Material.Get();
	}

	void MeshRenderer::SetMaterial(Material* material)
	{
		m_Material = material;
	}

	const AABB& MeshRenderer::GetBounds()
	{
		if (!m_Mesh.IsValid())
		{
			return m_Bounds;
		}

		Transform* transform = GetEntity()->GetTransform();
		size_t transformRecalculationFrame = transform->GetRecalculationFrame();
		if (m_RecalculationFrame != transformRecalculationFrame)
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
		BIND_FIELD(FieldInfo(TO_STRING(m_Material), &MeshRenderer::m_Material, BindingType::ObjectPtr).SetObjectType(Material::Type))
		END_OBJECT_BINDING()
	}
}
