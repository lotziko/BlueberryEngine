#include "bbpch.h"
#include "SceneArea.h"

#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"

#include "Editor\Preferences.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\EditorObjectManager.h"
#include "Editor\Selection.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Gizmos\GizmoRenderer.h"
#include "Editor\Gizmos\IconRenderer.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\DefaultRenderer.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Events\WindowEvents.h"

#include "SceneAreaMovement.h"

#include "imgui\imgui.h"
#include "imgui\imguizmo.h"

namespace Blueberry
{
	size_t SceneArea::s_SceneRedrawFrame = 0;

	SceneArea::SceneArea()
	{
		m_ColorRenderTarget = RenderTexture::Create(1920, 1080, 1, TextureFormat::R8G8B8A8_UNorm);
		m_DepthStencilRenderTarget = RenderTexture::Create(1920, 1080, 1, TextureFormat::D24_UNorm);

		m_GridMaterial = Material::Create((Shader*)AssetLoader::Load("assets/Grid.shader"));
		m_ObjectPicker = new SceneObjectPicker();

		// TODO save to config instead
		m_Position = Vector3(0, 10, 0);
		m_Rotation = Quaternion::CreateFromYawPitchRoll(0, ToRadians(-45), 0);

		Entity* cameraEntity = Object::Create<Entity>();
		cameraEntity->AddComponent<Transform>();
		m_Camera = cameraEntity->AddComponent<Camera>();
		cameraEntity->OnCreate();

		Selection::GetSelectionChanged().AddCallback<SceneArea, &SceneArea::RequestRedraw>(this);
		EditorSceneManager::GetSceneLoaded().AddCallback<SceneArea, &SceneArea::RequestRedraw>(this);
		WindowEvents::GetWindowFocused().AddCallback<SceneArea, &SceneArea::RequestRedraw>(this);
		EditorObjectManager::GetEntityCreated().AddCallback<SceneArea, &SceneArea::RequestRedraw>(this);
		EditorObjectManager::GetEntityDestroyed().AddCallback<SceneArea, &SceneArea::RequestRedraw>(this);
	}

	SceneArea::~SceneArea()
	{
		delete m_ColorRenderTarget;
		delete m_DepthStencilRenderTarget;
		delete m_ObjectPicker;

		Selection::GetSelectionChanged().RemoveCallback<SceneArea, &SceneArea::RequestRedraw>(this);
		EditorSceneManager::GetSceneLoaded().RemoveCallback<SceneArea, &SceneArea::RequestRedraw>(this);
		WindowEvents::GetWindowFocused().RemoveCallback<SceneArea, &SceneArea::RequestRedraw>(this);
		EditorObjectManager::GetEntityCreated().RemoveCallback<SceneArea, &SceneArea::RequestRedraw>(this);
		EditorObjectManager::GetEntityDestroyed().RemoveCallback<SceneArea, &SceneArea::RequestRedraw>(this);
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
		if (ImGui::Begin("Scene"))
		{
			if (ImGui::IsWindowAppearing())
			{
				RequestRedrawAll();
			}

			Camera::SetCurrent(m_Camera);

			ImGuiIO *io = &ImGui::GetIO();
			ImVec2 mousePos = ImGui::GetMousePos();
			ImVec2 pos = ImGui::GetCursorScreenPos();
			ImVec2 size = ImGui::GetContentRegionAvail();

			// Motion
			if (ImGui::IsWindowFocused())
			{
				Vector3 motion = GetMotion(m_Rotation);
				m_Position += motion;
				if (motion.LengthSquared() > 0)
				{
					RequestRedrawAll();
				}
			}

			// Zoom
			float mouseWheelDelta = io->MouseWheel;
			if (mouseWheelDelta != 0)
			{
				if (mousePos.x >= pos.x && mousePos.y >= pos.y && mousePos.x <= pos.x + size.x && mousePos.y <= pos.y + size.y)
				{
					SceneAreaMovement::HandleZoom(this, mouseWheelDelta, Vector2(mousePos.x - pos.x, size.y - (mousePos.y - pos.y)));
					RequestRedrawAll();
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
				RequestRedrawAll();
			}
			else
			{
				if (m_IsDragging)
				{
					m_IsDragging = false;
				}
			}

			// Prefab drop
			ImGui::Dummy(size);
			if (ImGui::BeginDragDropTarget())
			{
				const ImGuiPayload* payload = ImGui::GetDragDropPayload();
				if (payload != nullptr && payload->IsDataType("OBJECT_ID"))
				{
					Blueberry::ObjectId* id = (Blueberry::ObjectId*)payload->Data;
					Blueberry::Object* object = Blueberry::ObjectDB::GetObject(*id);

					if (object != nullptr && object->IsClassType(Entity::Type) && ImGui::AcceptDragDropPayload("OBJECT_ID"))
					{
						Scene* scene = EditorSceneManager::GetScene();
						if (scene != nullptr)
						{
							if (Blueberry::ObjectDB::HasGuid(object))
							{
								AssetLoader::Load(Blueberry::ObjectDB::GetGuidFromObject(object));
								scene->AddEntity(PrefabManager::CreateInstance((Entity*)object)->GetEntity());
							}
						}
					}
					RequestRedrawAll();
				}
				ImGui::EndDragDropTarget();
			}
			ImGui::SetCursorScreenPos(pos);

			// Selection
			if (ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered() && (!ImGuizmo::IsOver() || Selection::GetActiveObject() == nullptr))
			{
				if (ImGui::IsMouseClicked(0) && mousePos.x >= pos.x && mousePos.y >= pos.y && mousePos.x <= pos.x + size.x && mousePos.y <= pos.y + size.y)
				{
					Object* pickedObject = m_ObjectPicker->Pick(EditorSceneManager::GetScene(), m_Camera, (int)(mousePos.x - pos.x), (int)(mousePos.y - pos.y));
					if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
					{
						Selection::AddActiveObject(pickedObject);
					}
					else
					{
						Selection::SetActiveObject(pickedObject);
					}
				}
			}

			// Focus window on right mouse button down
			if (ImGui::IsWindowHovered() && ImGui::IsMouseDown(1))
			{
				ImGui::SetWindowFocus();
			}

			SetupCamera(size.x, size.y);

			if (s_SceneRedrawFrame >= Time::GetFrameCount())
			{
				DrawScene(size.x, size.y);

				m_ObjectPicker->DrawOutline(EditorSceneManager::GetScene(), m_Camera, m_ColorRenderTarget->Get());
			}
			DrawGizmos(Rectangle(pos.x, pos.y, size.x, size.y));

			ImGui::GetWindowDrawList()->AddImage(m_ColorRenderTarget->GetHandle(), ImVec2(pos.x, pos.y), ImVec2(pos.x + size.x, pos.y + size.y), ImVec2(0, 0), ImVec2(size.x / m_ColorRenderTarget->GetWidth(), size.y / m_ColorRenderTarget->GetHeight()));
			DrawControls();
		}
		ImGui::End();
	}

	float SceneArea::GetPerspectiveDistance(const float objectSize, const float fov)
	{
		return objectSize / sin(ToRadians(fov * 0.5f));
	}

	float SceneArea::GetCameraDistance()
	{
		if (m_Camera->IsOrthographic())
		{
			return m_Size * 2;
		}
		else
		{
			return GetPerspectiveDistance(m_Size, m_Camera->GetFieldOfView());
		}
	}

	Camera* SceneArea::GetCamera()
	{
		return m_Camera;
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

	void SceneArea::RequestRedrawAll()
	{
		s_SceneRedrawFrame = Time::GetFrameCount() + 1;
	}

	Vector3 SceneArea::GetCameraPosition()
	{
		// GetCameraDistance() is inverted because of right handed coordinate system
		return m_Position + Vector3::Transform(Vector3(0, 0, GetCameraDistance()), m_Camera->GetTransform()->GetRotation());
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
		if (m_Camera->GetAspectRatio() < 1.0f)
		{
			result /= m_Camera->GetAspectRatio();
		}
		return result;
	}

	void SceneArea::SetupCamera(const float& width, const float& height)
	{
		Transform* transform = m_Camera->GetTransform();
		// avoid changing this when there is no motion
		Quaternion rotation = GetCameraRotation();
		if (rotation != m_PreviousRotation)
		{
			transform->SetRotation(rotation);
			m_PreviousRotation = rotation;
		}
		Vector3 position = GetCameraPosition();
		if (position != m_PreviousPosition)
		{
			transform->SetPosition(position);
			m_PreviousPosition = position;
		}
		

		if (m_IsOrthographic)
		{
			m_Camera->SetOrthographic(true);
			m_Camera->SetOrthographicSize(GetCameraOrthographicSize());
		}
		else
		{
			m_Camera->SetOrthographic(false);
			m_Camera->SetFieldOfView(60);
		}
		Vector2 currentSize = m_Camera->GetPixelSize();
		if (currentSize.x != width || currentSize.y != height)
		{
			m_Camera->SetPixelSize(Vector2(width, height));
			RequestRedrawAll();
		}
	}

	void SceneArea::DrawControls()
	{
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.180f, 0.180f, 0.180f, 1.000f));
		ImVec2 cursor = ImGui::GetCursorPos();
		cursor.x += 5;
		cursor.y += 5;
		ImGui::SetCursorPos(cursor);

		ImGui::BeginGroup();

		if (EditorSceneManager::GetScene() != nullptr && !EditorSceneManager::IsRunning())
		{
			if (ImGui::Button("Save"))
			{
				EditorSceneManager::Save();
			}
			ImGui::SameLine();
		}

		if (Is2DMode())
		{
			if (ImGui::Button("3D"))
			{
				Set2DMode(false);
			}
			ImGui::SameLine();
		}
		else
		{
			if (ImGui::Button("2D"))
			{
				Set2DMode(true);
			}
			ImGui::SameLine();
		}

		// Snapping
		{
			const char* popId = "Snap";
			if (ImGui::Button("Snapping"))
			{
				ImGui::OpenPopup(popId);
			}
			ImGui::SameLine();

			if (ImGui::BeginPopup(popId))
			{
				ImGui::InputFloat("##positionSnap", Preferences::GetGizmoSnapping());
				ImGui::InputFloat("##rotationSnap", Preferences::GetGizmoSnapping() + 1);
				ImGui::InputFloat("##scaleSnap", Preferences::GetGizmoSnapping() + 2);
				ImGui::EndPopup();
			}

			if (ImGui::Button("Position"))
			{
				Preferences::SetGizmoOperation(ImGuizmo::OPERATION::TRANSLATE);
			}
			ImGui::SameLine();
			if (ImGui::Button("Rotation"))
			{
				Preferences::SetGizmoOperation(ImGuizmo::OPERATION::ROTATE);
			}
			ImGui::SameLine();
			if (ImGui::Button("Scale"))
			{
				Preferences::SetGizmoOperation(ImGuizmo::OPERATION::SCALE);
			}
		}

		ImGui::EndGroup();

		ImGui::PopStyleColor();
	}

	// TODO remove ImGuizmo and manually draw handles
	void SceneArea::DrawGizmos(const Rectangle& viewport)
	{
		ImDrawList* drawList = ImGui::GetForegroundDrawList();
		drawList->PushClipRect(ImVec2(viewport.x, viewport.y), ImVec2(viewport.x + viewport.width, viewport.y + viewport.height));
		ImGuizmo::SetDrawlist(drawList);
		ImGuizmo::SetRect(viewport.x, viewport.y, viewport.width, viewport.height);

		Scene* scene = EditorSceneManager::GetScene();
		if (scene != nullptr)
		{
			GfxDevice::SetRenderTarget(m_ColorRenderTarget->Get(), m_DepthStencilRenderTarget->Get());
			GfxDevice::SetViewport(0, 0, viewport.width, viewport.height);
			GizmoRenderer::Draw(scene, m_Camera);
			GfxDevice::SetRenderTarget(nullptr);
		}
		drawList->PopClipRect();
	}

	void SceneArea::DrawScene(const float width, const float height)
	{
		int viewportWidth = static_cast<int>(width);
		int viewportHeight = static_cast<int>(height);
		Color background = { 0.117f, 0.117f, 0.117f, 1 };

		Scene* scene = EditorSceneManager::GetScene();
		if (scene == nullptr)
		{
			GfxDevice::SetRenderTarget(m_ColorRenderTarget->Get());
			GfxDevice::ClearColor(background);
			GfxDevice::SetRenderTarget(nullptr);
			return;
		}
		DefaultRenderer::Draw(scene, m_Camera, Rectangle(0, 0, viewportWidth, viewportHeight), background, m_ColorRenderTarget, m_DepthStencilRenderTarget);
		GfxDevice::SetRenderTarget(m_ColorRenderTarget->Get(), m_DepthStencilRenderTarget->Get());
		GfxDevice::SetViewport(0, 0, viewportWidth, viewportHeight);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), m_GridMaterial));
		IconRenderer::Draw(scene, m_Camera);
		GfxDevice::SetRenderTarget(nullptr);
	}

	void SceneArea::LookAt(const Vector3& point, const Quaternion& direction, const float& newSize, const bool& isOrthographic)
	{
		m_Position = point;
		m_Rotation = direction;
		m_Size = newSize;
		m_IsOrthographic = isOrthographic;
	}

	void SceneArea::RequestRedraw()
	{
		RequestRedrawAll();
	}
}