#include "GameView.h"

#include "Editor\EditorLayer.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Menu\EditorMenuManager.h"

#include "Blueberry\Core\Application.h"
#include "Blueberry\Core\Screen.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Input\Cursor.h"
#include "Blueberry\Input\Input.h"
#include "Blueberry\Tools\CameraHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	OBJECT_DEFINITION(GameView, EditorWindow)
	{
		DEFINE_BASE_FIELDS(GameView, EditorWindow)
		EditorMenuManager::AddItem("Window/Game", &GameView::Open);
	}
	
	void GameView::Open()
	{
		EditorWindow* window = GetWindow(GameView::Type);
		window->SetTitle("Game");
		window->Show();
	}

	void GameView::OnDrawUI()
	{
		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();

		if (Application::IsRunning())
		{
			if (ImGui::IsWindowFocused())
			{
				Input::SetEnabled(true);
				Screen::SetAllowCursorLock(true);
				if (Cursor::IsHidden())
				{
					ImGui::SetMouseCursor(ImGuiMouseCursor_None);
				}
			}
			if (ImGui::IsKeyDown(ImGuiKey_Escape))
			{
				Input::SetEnabled(false);
				Screen::SetAllowCursorLock(false);
				ImGui::SetWindowFocus(nullptr);
			}
		}

		Scene* scene = EditorSceneManager::GetScene();
		if (scene != nullptr)
		{
			Camera* mainCamera = nullptr;
			for (Camera* camera : scene->GetComponents<Camera>())
			{
				if (camera->GetEntity()->IsActiveInHierarchy())
				{
					mainCamera = camera;
					break;
				}
			}
			if (mainCamera != nullptr)
			{
				EditorLayer::RequestFrameUpdate();
				Camera::SetCurrent(mainCamera);
				RectangleFloat viewport = CameraHelper::CalculateViewport(mainCamera, Rectangle(static_cast<long>(pos.x), static_cast<long>(pos.y), static_cast<long>(size.x), static_cast<long>(size.y)));
				Screen::SetGameViewport(Rectangle(static_cast<long>(viewport.x), static_cast<long>(viewport.y), static_cast<long>(viewport.width), static_cast<long>(viewport.height)));
				
				if (m_RenderTarget == nullptr || viewport.x != m_RenderTarget->GetWidth() || viewport.y != m_RenderTarget->GetHeight())
				{
					if (m_RenderTarget != nullptr)
					{
						GfxTexturePool::Release(m_RenderTarget);
					}
					m_RenderTarget = GfxTexturePool::Get(static_cast<uint32_t>(viewport.width), static_cast<uint32_t>(viewport.height), 1, TextureUsageFlags::RenderTarget, 1, 1, TextureFormat::R8G8B8A8_UNorm);
					mainCamera->SetPixelSize(Vector2(viewport.width, viewport.height));
				}

				DefaultRenderer::Draw(scene, mainCamera, Rectangle(0l, 0l, static_cast<long>(viewport.width), static_cast<long>(viewport.height)), m_RenderTarget);
				ImGui::GetWindowDrawList()->AddImage(reinterpret_cast<ImTextureID>(m_RenderTarget->GetHandle()), ImVec2(viewport.x, viewport.y), ImVec2(viewport.x + viewport.width, viewport.y + viewport.height), ImVec2(0.0f, 0.0f), ImVec2(viewport.width / m_RenderTarget->GetWidth(), viewport.height / m_RenderTarget->GetHeight()));
			}
		}
	}
}
