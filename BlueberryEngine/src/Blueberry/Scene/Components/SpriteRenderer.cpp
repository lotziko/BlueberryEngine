#include "bbpch.h"
#include "SpriteRenderer.h"

#include "Blueberry\Core\GlobalServices.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Renderer, SpriteRenderer)
	COMPONENT_DEFINITION(SpriteRenderer)

	SpriteRenderer::SpriteRenderer()
	{
		m_Color = Color(1, 1, 1, 1);
		g_AssetManager->Load<Texture>("assets/TestImage", m_Texture);
	}
}