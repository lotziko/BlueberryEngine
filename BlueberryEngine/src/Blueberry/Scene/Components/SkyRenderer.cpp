#include "Blueberry\Scene\Components\SkyRenderer.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Graphics\DefaultTextures.h"

namespace Blueberry
{
	OBJECT_DEFINITION(SkyRenderer, Component)
	{
		DEFINE_BASE_FIELDS(SkyRenderer, Component)
		DEFINE_FIELD(SkyRenderer, m_Material, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Material::Type))
		DEFINE_FIELD(SkyRenderer, m_AmbientColor, BindingType::Color, FieldOptions())
		DEFINE_ITERATOR(SkyRenderer)
	}

	Material* SkyRenderer::GetMaterial() const
	{
		return m_Material.Get();
	}

	const Color& SkyRenderer::GetAmbientColor() const
	{
		return m_AmbientColor;
	}

	void SkyRenderer::SetAmbientColor(const Color& color)
	{
		m_AmbientColor = color;
	}
}
