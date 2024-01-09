#include "bbpch.h"
#include "SceneArea.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\SceneRenderer.h"
#include "Blueberry\Graphics\GfxTexture.h"

#include "Editor\EditorSceneManager.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\StandardMeshes.h"

#include "Editor\EditorMaterials.h"
#include "SceneAreaMovement.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	SceneArea::SceneArea()
	{
		TextureProperties properties;
		properties.width = 1920;
		properties.height = 1080;
		properties.data = nullptr;
		properties.type = TextureType::RenderTarget;
		g_GraphicsDevice->CreateTexture(properties, m_SceneRenderTarget);
	}

	void SceneArea::DrawUI()
	{
		ImGui::Begin("Scene");
		
		if (EditorSceneManager::GetScene() != nullptr)
		{
			if (ImGui::Button("Save"))
			{
				EditorSceneManager::Save();
			}
		}
		
		if (Is2DMode())
		{
			if (ImGui::Button("3D"))
			{
				Set2DMode(false);
			}
		}
		else
		{
			if (ImGui::Button("2D"))
			{
				Set2DMode(true);
			}
		}

		ImGuiIO *io = &ImGui::GetIO();
		ImVec2 mousePos = ImGui::GetMousePos();
		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();
		
		if (ImGui::IsKeyDown(ImGuiKey_D))
		{
			m_Position += Vector3::Transform(Vector3::Right, m_Rotation);
		}
		else if (ImGui::IsKeyDown(ImGuiKey_A))
		{
			m_Position += Vector3::Transform(Vector3::Left, m_Rotation);
		}
		if (ImGui::IsKeyDown(ImGuiKey_E))
		{
			m_Position += Vector3::Transform(Vector3::Up, m_Rotation);
		}
		else if (ImGui::IsKeyDown(ImGuiKey_Q))
		{
			m_Position += Vector3::Transform(Vector3::Down, m_Rotation);
		}
		if (ImGui::IsKeyDown(ImGuiKey_W))
		{
			m_Position += Vector3::Transform(Vector3::Forward, m_Rotation);
		}
		else if (ImGui::IsKeyDown(ImGuiKey_S))
		{
			m_Position += Vector3::Transform(Vector3::Backward, m_Rotation);
		}

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

		// Selection
		if (ImGui::IsMouseClicked(0) && mousePos.x >= pos.x && mousePos.y >= pos.y && mousePos.x <= pos.x + size.x && mousePos.y <= pos.y + size.y)
		{
			m_ObjectPicker.Pick(EditorSceneManager::GetScene(), m_Camera, (int)(mousePos.x - pos.x), (int)(mousePos.y - pos.y), size.x, size.y);
		}

		//SceneAreaMovement::HandleDrag(this, Vector2(10, 10));

		SetupCamera(size.x, size.y);
		DrawScene(size.x, size.y);

		ImGui::GetWindowDrawList()->AddImage(m_SceneRenderTarget->GetHandle(), ImVec2(pos.x, pos.y), ImVec2(pos.x + size.x, pos.y + size.y), ImVec2(0, 0), ImVec2(size.x / m_SceneRenderTarget->GetWidth(), size.y / m_SceneRenderTarget->GetHeight()));

		ImGui::End();
	}

	float SceneArea::GetPerspectiveDistance(const float objectSize, const float fov)
	{
		return objectSize / sin(ToRadians(fov * 0.5f));
	}

	float SceneArea::GetCameraDistance()
	{
		if (m_Camera.IsOrthographic())
		{
			return m_Size * 2;
		}
		else
		{
			return GetPerspectiveDistance(m_Size, m_Camera.GetFieldOfView());
		}
	}

	BaseCamera* SceneArea::GetCamera()
	{
		return &m_Camera;
	}

	Vector3 SceneArea::GetPosition()
	{
		return m_Position;
	}

	void SceneArea::SetPosition(const Vector3& position)
	{
		m_Position = position;
	}

	Quaternion SceneArea::GetRotation()
	{
		return m_Rotation;
	}

	void SceneArea::SetRotation(const Quaternion& rotation)
	{
		m_Rotation = rotation;
	}

	float SceneArea::GetSize()
	{
		return m_Size;
	}

	void SceneArea::SetSize(const float& size)
	{
		m_Size = size;
	}

	bool SceneArea::IsOrthographic()
	{
		return m_IsOrthographic;
	}

	bool SceneArea::Is2DMode()
	{
		return m_Is2DMode;
	}

	void SceneArea::Set2DMode(const bool& is2DMode)
	{
		m_Is2DMode = is2DMode;
		if (m_Is2DMode)
		{
			LookAt(m_Position, Quaternion::Identity, m_Size, true);
		}
		else
		{
			LookAt(m_Position, Quaternion::Identity, m_Size, false);
		}
	}

	Vector3 SceneArea::GetCameraPosition()
	{
		// GetCameraDistance() is inverted because of right handed coordinate system
		return m_Position + Vector3::Transform(Vector3(0, 0, GetCameraDistance()), m_Camera.GetRotation());
	}

	Quaternion SceneArea::GetCameraRotation()
	{
		return m_Is2DMode ? Quaternion::Identity : m_Rotation;
	}

	Vector3 SceneArea::GetCameraTargetPosition()
	{
		// GetCameraDistance() is inverted because of right handed coordinate system
		return m_Position + Vector3::Transform(Vector3(0, 0, GetCameraDistance()), m_Rotation);
	}

	Quaternion SceneArea::GetCameraTargetRotation()
	{
		return m_Rotation;
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

	void SceneArea::SetupCamera(const float& width, const float& height)
	{
		m_Camera.SetRotation(GetCameraRotation());
		m_Camera.SetPosition(GetCameraPosition());

		if (m_IsOrthographic)
		{
			m_Camera.SetOrthographic(true);
			m_Camera.SetOrthographicSize(GetCameraOrthographicSize());
			m_Camera.SetPixelSize(Vector2(width, height));
		}
		else
		{
			m_Camera.SetOrthographic(false);
			m_Camera.SetFieldOfView(60);
		}
	}

	void SceneArea::DrawScene(const float width, const float height)
	{
		g_GraphicsDevice->SetRenderTarget(m_SceneRenderTarget);
		g_GraphicsDevice->SetViewport(0, 0, static_cast<int>(width), static_cast<int>(height));
		g_GraphicsDevice->ClearColor({ 0.117f, 0.117f, 0.117f, 1 });

		Scene* scene = EditorSceneManager::GetScene();
		if (scene != nullptr)
		{
			SceneRenderer::Draw(scene, m_Camera.GetViewMatrix(), m_Camera.GetProjectionMatrix());
		}
		g_GraphicsDevice->SetRenderTarget(nullptr);
	}

	void SceneArea::LookAt(const Vector3& point, const Quaternion& direction, const float& newSize, const bool& isOrthographic)
	{
		m_Position = point;
		m_Rotation = direction;
		m_Size = newSize;
		m_IsOrthographic = isOrthographic;
	}
}