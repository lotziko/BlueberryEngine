#include "GameView.h"

#include "Editor\EditorLayer.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Menu\EditorMenuManager.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxRenderTexturePool.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Input\Cursor.h"

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

		if (ImGui::IsWindowFocused())
		{
			Screen::SetAllowCursorLock(true);
			Screen::SetGameViewport(Rectangle(0, 0, size.x, size.y));
			if (Cursor::IsHidden())
			{
				ImGui::SetMouseCursor(ImGuiMouseCursor_None);
			}
		}
		if (ImGui::IsKeyDown(ImGuiKey_Escape))
		{
			Screen::SetAllowCursorLock(false);
			ImGui::SetWindowFocus(nullptr);
		}

		Scene* scene = EditorSceneManager::GetScene();
		if (scene != nullptr)
		{
			Camera* camera = nullptr;
			for (auto& pair : scene->GetIterator<Camera>())
			{
				if (pair.second->GetEntity()->IsActiveInHierarchy())
				{
					camera = static_cast<Camera*>(pair.second);
					break;
				}
			}
			if (camera != nullptr)
			{
				EditorLayer::RequestFrameUpdate();
				Camera::SetCurrent(camera);

				float areaAspectRatio = size.x / size.y;
				float cameraAspectRatio = camera->GetAspectRatio();

				float x, y, width, height;

				if (areaAspectRatio > cameraAspectRatio)
				{
					width = size.y * cameraAspectRatio;
					x = pos.x + (size.x - width) / 2.0f;
					y = pos.y;
					height = size.y;
				}
				else
				{
					height = size.x / cameraAspectRatio;
					y = pos.y + (size.y - height) / 2.0f;
					x = pos.x;
					width = size.x;
				}

				// TODO viewport change
				Vector2 viewport = Vector2(size.x, size.y);

				if (m_RenderTarget == nullptr || viewport.x != m_RenderTarget->GetWidth() || viewport.y != m_RenderTarget->GetHeight())
				{
					if (m_RenderTarget != nullptr)
					{
						GfxRenderTexturePool::Release(m_RenderTarget);
					}
					m_RenderTarget = GfxRenderTexturePool::Get(viewport.x, viewport.y, 1, 1, TextureFormat::R8G8B8A8_UNorm);
					camera->SetPixelSize(Vector2(width, height));
				}

				DefaultRenderer::Draw(scene, camera, Rectangle(0, 0, viewport.x, viewport.y), Color(0, 0, 0, 1), m_RenderTarget);
				ImGui::GetWindowDrawList()->AddImage(reinterpret_cast<ImTextureID>(m_RenderTarget->GetHandle()), ImVec2(x, y), ImVec2(x + width, y + height), ImVec2(0, 0), ImVec2(viewport.x / m_RenderTarget->GetWidth(), viewport.y / m_RenderTarget->GetHeight()));
			}
		}
	}
}
