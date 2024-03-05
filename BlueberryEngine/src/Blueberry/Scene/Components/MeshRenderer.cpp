#include "bbpch.h"
#include "MeshRenderer.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Entity.h"

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

	void MeshRenderer::BindProperties()
	{
		BEGIN_OBJECT_BINDING(MeshRenderer)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &MeshRenderer::m_Entity, BindingType::ObjectPtr, Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Mesh), &MeshRenderer::m_Mesh, BindingType::ObjectPtr, Mesh::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Material), &MeshRenderer::m_Material, BindingType::ObjectPtr, Material::Type))
		END_OBJECT_BINDING()
	}
}
