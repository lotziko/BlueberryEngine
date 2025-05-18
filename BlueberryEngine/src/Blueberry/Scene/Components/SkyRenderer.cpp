#include "Blueberry\Scene\Components\SkyRenderer.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(SkyRenderer, Component)
	{
		DEFINE_BASE_FIELDS(SkyRenderer, Component)
		DEFINE_FIELD(SkyRenderer, m_Material, BindingType::ObjectPtr, FieldOptions().SetObjectType(Material::Type))
	}

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
}
