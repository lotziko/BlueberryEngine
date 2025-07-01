#include "Blueberry\Scene\Components\SkyRenderer.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Core\ClassDB.h"

#include "..\..\Graphics\DefaultTextures.h"

namespace Blueberry
{
	OBJECT_DEFINITION(SkyRenderer, Component)
	{
		DEFINE_BASE_FIELDS(SkyRenderer, Component)
		DEFINE_FIELD(SkyRenderer, m_Material, BindingType::ObjectPtr, FieldOptions().SetObjectType(Material::Type))
		DEFINE_FIELD(SkyRenderer, m_AmbientColor, BindingType::Color, {})
		DEFINE_FIELD(SkyRenderer, m_ReflectionTexture, BindingType::ObjectPtr, FieldOptions().SetObjectType(TextureCube::Type))
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

	const Color& SkyRenderer::GetAmbientColor()
	{
		return m_AmbientColor;
	}

	void SkyRenderer::SetAmbientColor(const Color& color)
	{
		m_AmbientColor = color;
	}

	TextureCube* SkyRenderer::GetReflectionTexture()
	{
		if (!m_ReflectionTexture.IsValid())
		{
			return DefaultTextures::GetBlackCube();
		}
		return m_ReflectionTexture.Get();
	}

	void SkyRenderer::SetReflectionTexture(TextureCube* texture)
	{
		m_ReflectionTexture = texture;
	}
}
