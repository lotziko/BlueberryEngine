#include "bbpch.h"
#include "GameView.h"

#include "Editor\EditorLayer.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Menu\EditorMenuManager.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Graphics\DefaultRenderer.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Scene\Scene.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	OBJECT_DEFINITION(EditorWindow, GameView)
	
	void GameView::Open()
	{
		EditorWindow* window = GetWindow(GameView::Type);
		window->SetTitle("Game");
		window->Show();
	}

	void GameView::BindProperties()
	{
		BEGIN_OBJECT_BINDING(GameView)
		BIND_FIELD(FieldInfo(TO_STRING(m_Title), &GameView::m_Title, BindingType::String))
		BIND_FIELD(FieldInfo(TO_STRING(m_RawData), &GameView::m_RawData, BindingType::ByteData))
		END_OBJECT_BINDING()

		EditorMenuManager::AddItem("Window/Game", &GameView::Open);
	}

	void GameView::OnDrawUI()
	{
		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();

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
						Object::Destroy(m_RenderTarget);
					}
					m_RenderTarget = RenderTexture::Create(viewport.x, viewport.y, 1, 1, TextureFormat::R8G8B8A8_UNorm);
					camera->SetPixelSize(Vector2(width, height));
				}

				DefaultRenderer::Draw(scene, camera, Rectangle(0, 0, viewport.x, viewport.y), Color(0, 0, 0, 1), m_RenderTarget);
				ImGui::GetWindowDrawList()->AddImage(reinterpret_cast<ImTextureID>(m_RenderTarget->GetHandle()), ImVec2(x, y), ImVec2(x + width, y + height), ImVec2(0, 0), ImVec2(viewport.x / m_RenderTarget->GetWidth(), viewport.y / m_RenderTarget->GetHeight()));
			}
		}
	}
}
