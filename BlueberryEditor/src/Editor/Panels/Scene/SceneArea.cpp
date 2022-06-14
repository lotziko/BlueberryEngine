#include "bbpch.h"
#include "SceneArea.h"

#include "Blueberry\Core\ServiceContainer.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\SceneRenderer.h"

SceneArea::SceneArea(const Ref<Scene>& scene) : m_Scene(scene)
{
}

void SceneArea::Draw()
{
	ServiceContainer::GraphicsDevice->SetViewport(m_Viewport.x, m_Viewport.y, m_Viewport.width, m_Viewport.height);
	ServiceContainer::GraphicsDevice->ClearColor({ 0, 0, 0, 1 });

	if (m_Viewport.width > 0)
	{
		m_Camera->SetResolution(Vector2(m_Viewport.width, m_Viewport.height));
	}

	if (m_Scene != NULL)
	{
		SceneRenderer::Draw(m_Scene);
	}
}