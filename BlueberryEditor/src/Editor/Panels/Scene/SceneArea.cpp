#include "bbpch.h"
#include "SceneArea.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\SceneRenderer.h"

namespace Blueberry
{
	SceneArea::SceneArea(const Ref<Scene>& scene) : m_Scene(scene)
	{
	}

	void SceneArea::Draw()
	{
		g_GraphicsDevice->SetViewport(static_cast<int>(m_Viewport.x), static_cast<int>(m_Viewport.y), static_cast<int>(m_Viewport.width), static_cast<int>(m_Viewport.height));
		g_GraphicsDevice->ClearColor({ 0, 0, 0, 1 });

		if (m_Viewport.width > 0)
		{
			m_Camera.SetViewport(m_Viewport);
			m_Camera.Update();
		}

		if (m_Scene != NULL)
		{
			SceneRenderer::Draw(m_Scene, m_Camera.GetViewMatrix(), m_Camera.GetProjectionMatrix());
		}
	}
}