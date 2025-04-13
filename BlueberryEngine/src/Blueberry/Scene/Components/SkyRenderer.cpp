#include "bbpch.h"
#include "SkyRenderer.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Graphics\Material.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, SkyRenderer)

	void SkyRenderer::OnEnable()
	{
		AddToSceneComponents(SkyRenderer::Type);
	}

	void SkyRenderer::OnDisable()
	{
		RemoveFromSceneComponents(SkyRenderer::Type);
	}

	Material* SkyRenderer::GetMaterial()
	{
		return m_Material.Get();
	}

	void SkyRenderer::BindProperties()
	{
		BEGIN_OBJECT_BINDING(SkyRenderer)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &SkyRenderer::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Material), &SkyRenderer::m_Material, BindingType::ObjectPtr).SetObjectType(Material::Type))
		END_OBJECT_BINDING()
	}
}
