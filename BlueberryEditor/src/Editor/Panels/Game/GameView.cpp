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
					DefaultRenderer::Draw(scene, camera, Rectangle(0, 0, size.x, size.y), Color(0, 0, 0, 1), m_RenderTarget);
					ImGui::GetWindowDrawList()->AddImage(m_RenderTarget->GetHandle(), ImVec2(pos.x, pos.y), ImVec2(pos.x + size.x, pos.y + size.y), ImVec2(0, 0), ImVec2(size.x / m_RenderTarget->GetWidth(), size.y / m_RenderTarget->GetHeight()));
				}
			}
		}
		ImGui::End();
	}
}
