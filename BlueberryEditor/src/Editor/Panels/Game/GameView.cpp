#include "bbpch.h"
#include "GameView.h"

#include "Editor\EditorSceneManager.h"

#include "Blueberry\Graphics\DefaultRenderer.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Scene\Scene.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	GameView::GameView()
	{
		m_RenderTarget = RenderTexture::Create(1920, 1080, 1, TextureFormat::R8G8B8A8_UNorm);
	}

	GameView::~GameView()
	{
		delete m_RenderTarget;
	}

	void GameView::DrawUI()
	{
		if (ImGui::Begin("Game"))
		{
			ImVec2 pos = ImGui::GetCursorScreenPos();
			ImVec2 size = ImGui::GetContentRegionAvail();

			Scene* scene = EditorSceneManager::GetScene();
			if (scene != nullptr)
			{
				Camera* camera = nullptr;
				for (auto& component : scene->GetIterator<Camera>())
				{
					camera = static_cast<Camera*>(component.second);
					break;
				}
				if (camera != nullptr)
				{
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
					
					DefaultRenderer::Draw(scene, camera, Rectangle(0, 0, viewport.x, viewport.y), Color(0, 0, 0, 1), m_RenderTarget);
					ImGui::GetWindowDrawList()->AddImage(m_RenderTarget->GetHandle(), ImVec2(x, y), ImVec2(x + width, y + height), ImVec2(0, 0), ImVec2(viewport.x / m_RenderTarget->GetWidth(), viewport.y / m_RenderTarget->GetHeight()));
				}
			}
		}
		ImGui::End();
	}
}