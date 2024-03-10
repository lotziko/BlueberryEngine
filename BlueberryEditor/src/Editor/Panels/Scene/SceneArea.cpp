#include "bbpch.h"
#include "SceneArea.h"

#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\SceneRenderer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"

#include "Editor\EditorSceneManager.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Assets\AssetLoader.h"

#include "SceneAreaMovement.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	SceneArea::SceneArea()
	{
		TextureProperties properties = {};
		properties.width = 1920;
		properties.height = 1080;
		properties.isRenderTarget = true;
		properties.format = TextureFormat::R8G8B8A8_UNorm;
		GfxDevice::CreateTexture(properties, m_SceneRenderTarget);

		properties.format = TextureFormat::D24_UNorm;
		GfxDevice::CreateTexture(properties, m_SceneDepthStencil);

		m_GridMaterial = Material::Create((Shader*)AssetLoader::Load("assets/Grid.shader"));
		m_ObjectPicker = new SceneObjectPicker(m_SceneDepthStencil);

		// TODO save to config instead
		m_Position = Vector3(0, 10, 0);
		m_Rotation = Quaternion::CreateFromYawPitchRoll(0, ToRadians(-45), 0);
	}

	SceneArea::~SceneArea()
	{
		delete m_SceneRenderTarget;
		delete m_SceneDepthStencil;
		delete m_ObjectPicker;
	}

	Vector3 GetMotion(const Quaternion& rotation)
	{
		Vector3 motion = Vector3::Zero;
		if (ImGui::IsKeyDown(ImGuiKey_D))
		{
			motion += Vector3::Transform(Vector3::Right, rotation);
		}
		else if (ImGui::IsKeyDown(ImGuiKey_A))
		{
			motion += Vector3::Transform(Vector3::Left, rotation);
		}
		if (ImGui::IsKeyDown(ImGuiKey_E))
		{
			motion += Vector3::Transform(Vector3::Up, rotation);
		}
		else if (ImGui::IsKeyDown(ImGuiKey_Q))
		{
			motion += Vector3::Transform(Vector3::Down, rotation);
		}
		if (ImGui::IsKeyDown(ImGuiKey_W))
		{
			motion += Vector3::Transform(Vector3::Forward, rotation);
		}
		else if (ImGui::IsKeyDown(ImGuiKey_S))
		{
			motion += Vector3::Transform(Vector3::Backward, rotation);
		}

		if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
		{
			motion *= 1;
		}
		else
		{
			motion *= 0.25;
		}
		return motion;
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
		
		// Motion
		m_Position += GetMotion(m_Rotation);

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
			m_ObjectPicker->Pick(EditorSceneManager::GetScene(), m_Camera, (int)(mousePos.x - pos.x), (int)(mousePos.y - pos.y));
		}

		SetupCamera(size.x, size.y);
		DrawScene(size.x, size.y);
		m_ObjectPicker->DrawOutline(EditorSceneManager::GetScene(), m_Camera, m_SceneRenderTarget);

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
		}
		else
		{
			m_Camera.SetOrthographic(false);
			m_Camera.SetFieldOfView(60);
		}
		m_Camera.SetPixelSize(Vector2(width, height));
	}

	void SceneArea::DrawScene(const float width, const float height)
	{
		GfxDevice::SetRenderTarget(m_SceneRenderTarget, m_SceneDepthStencil);
		GfxDevice::SetViewport(0, 0, static_cast<int>(width), static_cast<int>(height));
		GfxDevice::ClearColor({ 0.117f, 0.117f, 0.117f, 1 });
		GfxDevice::ClearDepth(1.0f);

		GfxDevice::SetSurfaceType(SurfaceType_Opaque);
		GfxDevice::SetCullMode(CullMode_Front);
		Scene* scene = EditorSceneManager::GetScene();
		if (scene != nullptr)
		{
			SceneRenderer::Draw(scene, &m_Camera);
		}
		
		GfxDevice::SetCullMode(CullMode_None);
		GfxDevice::SetSurfaceType(SurfaceType_DepthTransparent); 
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), m_GridMaterial));
		GfxDevice::SetRenderTarget(nullptr);
	}

	void SceneArea::LookAt(const Vector3& point, const Quaternion& direction, const float& newSize, const bool& isOrthographic)
	{
		m_Position = point;
		m_Rotation = direction;
		m_Size = newSize;
		m_IsOrthographic = isOrthographic;
	}
}