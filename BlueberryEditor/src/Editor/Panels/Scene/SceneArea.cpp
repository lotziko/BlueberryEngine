#include "bbpch.h"
#include "SceneArea.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\SceneRenderer.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	SceneArea::SceneArea(const Ref<Scene>& scene) : m_Scene(scene)
	{
		g_GraphicsDevice->CreateRenderTarget(1920, 1080, m_SceneRenderTarget);
	}

	void SceneArea::DrawUI()
	{
		ImGui::Begin("Scene");
		
		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();

		DrawScene(size.x, size.y);

		ImGui::GetWindowDrawList()->AddImage(m_SceneRenderTarget->GetHandle(), ImVec2(pos.x, pos.y), ImVec2(pos.x + size.x, pos.y + size.y), ImVec2(0, 0), ImVec2(size.x / m_SceneRenderTarget->GetWidth(), size.y / m_SceneRenderTarget->GetHeight()));

		ImGui::End();
	}

	void SceneArea::DrawScene(const float width, const float height)
	{
		g_GraphicsDevice->BindRenderTarget(m_SceneRenderTarget);
		g_GraphicsDevice->SetViewport(0, 0, static_cast<int>(width), static_cast<int>(height));
		g_GraphicsDevice->ClearColor({ 0, 0, 0, 1 });

		if (width > 0)
		{
			m_Camera.SetViewport({ 0, 0, width, height });
			m_Camera.Update();
		}

		if (m_Scene != NULL)
		{
			SceneRenderer::Draw(m_Scene, m_Camera.GetViewMatrix(), m_Camera.GetProjectionMatrix());
		}
		g_GraphicsDevice->UnbindRenderTarget();
	}
}