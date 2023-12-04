#include "bbpch.h"
#include "SceneArea.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\SceneRenderer.h"
#include "Blueberry\Graphics\GfxTexture.h"

#include "Editor\EditorSceneManager.h"

#include "Blueberry\Graphics\StandardMeshes.h"
#include "Editor\EditorMaterials.h"
#include "SceneAreaMovement.h"

#include "Blueberry\Serialization\YamlHelper.h"
#include "Blueberry\Scene\Scene.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	SceneArea::SceneArea()
	{
		TextureProperties properties;
		properties.width = 1920;
		properties.height = 1080;
		properties.data = nullptr;
		properties.isRenderTarget = true;
		g_GraphicsDevice->CreateTexture(properties, m_SceneRenderTarget);
	}

	void SceneArea::DrawUI()
	{
		ImGui::Begin("Scene");
		DrawSceneSave();
		
		ImGuiIO *io = &ImGui::GetIO();
		ImVec2 mousePos = ImGui::GetMousePos();
		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();
		
		m_Camera.SetPixelSize(Vector2(size.x, size.y));
		m_Camera.SetOrthographicSize(GetCameraOrthographicSize());
		m_Camera.SetPixelsPerUnit(32);
		m_Camera.SetPosition(GetCameraPosition());

		// Zoom
		float mouseWheelDelta = io->MouseWheel;
		if (mouseWheelDelta != 0)
		{
			if (mousePos.x >= pos.x && mousePos.y >= pos.y && mousePos.x <= pos.x + size.x && mousePos.y <= pos.y + size.y)
			{
				SceneAreaMovement::HandleZoom(this, mouseWheelDelta, Vector2(mousePos.x - pos.x, size.y - (mousePos.y - pos.y)));
			}
		}

		// Dragging
		if (ImGui::IsMouseDragging(1, 0))
		{
			if (m_IsDragging == false)
			{
				ImVec2 clickPos = io->MouseClickedPos[1];
				if (clickPos.x >= pos.x && clickPos.y >= pos.y && clickPos.x <= pos.x + size.x && clickPos.y <= pos.y + size.y)
				{
					m_IsDragging = true;
					m_PreviousDragDelta = Vector2::Zero;
				}
			}
			else
			{
				ImVec2 dragDelta = ImGui::GetMouseDragDelta(1, 0);
				SceneAreaMovement::HandleDrag(this, Vector2(dragDelta.x - m_PreviousDragDelta.x, dragDelta.y - m_PreviousDragDelta.y));
				m_PreviousDragDelta = Vector2(dragDelta.x, dragDelta.y);
			}
		}
		else
		{
			if (m_IsDragging)
			{
				m_IsDragging = false;
			}
		}

		DrawScene(size.x, size.y);

		ImGui::GetWindowDrawList()->AddImage(m_SceneRenderTarget->GetHandle(), ImVec2(pos.x, pos.y), ImVec2(pos.x + size.x, pos.y + size.y), ImVec2(0, 0), ImVec2(size.x / m_SceneRenderTarget->GetWidth(), size.y / m_SceneRenderTarget->GetHeight()));

		ImGui::End();
	}

	float SceneArea::GetCameraDistance()
	{
		return m_Size * 2;
	}

	Vector3 SceneArea::GetCameraPosition()
	{
		return m_Position + Vector3::Transform(Vector3(0, 0, -m_Size * 2), m_Camera.GetRotation());
	}

	float SceneArea::GetCameraOrthographicSize()
	{
		float result = m_Size;
		if (m_Camera.GetAspectRatio() < 1.0f)
		{
			result /= m_Camera.GetAspectRatio();
		}
		return result;
	}

	void SceneArea::DrawSceneSave()
	{
		if (ImGui::Button("Create"))
		{
			EditorSceneManager::CreateEmpty("");
		}

		if (ImGui::Button("Save"))
		{
			EditorSceneManager::Save();
		}

		if (ImGui::Button("Load"))
		{
			EditorSceneManager::Load("");
		}
	}

	void SceneArea::DrawScene(const float width, const float height)
	{
		g_GraphicsDevice->SetRenderTarget(m_SceneRenderTarget.get());
		g_GraphicsDevice->SetViewport(0, 0, static_cast<int>(width), static_cast<int>(height));
		g_GraphicsDevice->ClearColor({ 0.117f, 0.117f, 0.117f, 1 });
		//g_GraphicsDevice->Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), EditorMaterials::GetEditorGridMaterial()));

		Ref<Scene> scene = EditorSceneManager::GetScene();
		if (scene != nullptr)
		{
			SceneRenderer::Draw(scene, m_Camera.GetViewMatrix(), m_Camera.GetProjectionMatrix());
		}
		g_GraphicsDevice->SetRenderTarget(nullptr);
	}
}