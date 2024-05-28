#include "bbpch.h"
#include "SceneArea.h"

#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Graphics\SceneRenderer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"

#include "Editor\EditorSceneManager.h"
#include "Editor\Selection.h"

#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Assets\AssetLoader.h"

#include "SceneAreaMovement.h"

#include "imgui\imgui.h"
#include "imgui\imguizmo.h"

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
		
		BaseCamera::SetCurrent(&m_Camera);

		ImGuiIO *io = &ImGui::GetIO();
		ImVec2 mousePos = ImGui::GetMousePos();
		ImVec2 pos = ImGui::GetCursorScreenPos();
		ImVec2 size = ImGui::GetContentRegionAvail();
		
		// Motion
		if (ImGui::IsWindowFocused())
		{
			m_Position += GetMotion(m_Rotation);
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
						AssetLoader::Load(Blueberry::ObjectDB::GetGuidFromObject(object));
						scene->AddEntity(static_cast<Entity*>(Object::Clone(object)));
					}
				}
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
		DrawScene(size.x, size.y);
		m_ObjectPicker->DrawOutline(EditorSceneManager::GetScene(), m_Camera, m_SceneRenderTarget);

		ImGui::GetWindowDrawList()->AddImage(m_SceneRenderTarget->GetHandle(), ImVec2(pos.x, pos.y), ImVec2(pos.x + size.x, pos.y + size.y), ImVec2(0, 0), ImVec2(size.x / m_SceneRenderTarget->GetWidth(), size.y / m_SceneRenderTarget->GetHeight()));
		DrawGizmos(Rectangle(pos.x, pos.y, size.x, size.y));
		DrawControls();

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

	void SceneArea::DrawControls()
	{
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.180f, 0.180f, 0.180f, 1.000f));
		ImVec2 cursor = ImGui::GetCursorPos();
		cursor.x += 5;
		cursor.y += 5;
		ImGui::SetCursorPos(cursor);

		ImGui::BeginGroup();

		if (EditorSceneManager::GetScene() != nullptr)
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
				ImGui::InputFloat("##positionSnap", &m_GizmoSnapping[0]);
				ImGui::InputFloat("##rotationSnap", &m_GizmoSnapping[1]);
				ImGui::InputFloat("##scaleSnap", &m_GizmoSnapping[2]);
				ImGui::EndPopup();
			}

			if (ImGui::Button("Position"))
			{
				m_GizmoOperation = ImGuizmo::OPERATION::TRANSLATE;
			}
			ImGui::SameLine();
			if (ImGui::Button("Rotation"))
			{
				m_GizmoOperation = ImGuizmo::OPERATION::ROTATE;
			}
			ImGui::SameLine();
			if (ImGui::Button("Scale"))
			{
				m_GizmoOperation = ImGuizmo::OPERATION::SCALE;
			}
		}

		ImGui::EndGroup();

		ImGui::PopStyleColor();
	}

	void SceneArea::DrawGizmos(const Rectangle& viewport)
	{
		Object* selectedObject = Selection::GetActiveObject();

		if (selectedObject != nullptr)
		{
			if (selectedObject->GetType() == Entity::Type)
			{
				Transform* transform = ((Entity*)selectedObject)->GetTransform();
				Matrix transformMatrix = Matrix::CreateScale(transform->GetLocalScale()) * Matrix::CreateFromQuaternion(transform->GetLocalRotation()) * Matrix::CreateTranslation(transform->GetLocalPosition());
				Transform* parentTransform = transform->GetParent();
				if (parentTransform != nullptr)
				{
					transformMatrix *= parentTransform->GetLocalToWorldMatrix();
				}

				ImGuizmo::SetDrawlist();
				ImGuizmo::SetRect(viewport.x, viewport.y, viewport.width, viewport.height);
				BaseCamera* camera = BaseCamera::GetCurrent();
				if (camera != nullptr)
				{
					float snapping[3];
					if (m_GizmoOperation == ImGuizmo::OPERATION::TRANSLATE)
					{
						snapping[0] = m_GizmoSnapping[0];
						snapping[1] = m_GizmoSnapping[0];
						snapping[2] = m_GizmoSnapping[0];
					}
					else if (m_GizmoOperation == ImGuizmo::OPERATION::ROTATE)
					{
						snapping[0] = m_GizmoSnapping[1];
						snapping[1] = m_GizmoSnapping[1];
						snapping[2] = m_GizmoSnapping[1];
					}
					else if (m_GizmoOperation == ImGuizmo::OPERATION::SCALE)
					{
						snapping[0] = m_GizmoSnapping[2];
						snapping[1] = m_GizmoSnapping[2];
						snapping[2] = m_GizmoSnapping[2];
					}

					if (ImGuizmo::Manipulate((float*)camera->GetViewMatrix().m, (float*)camera->GetProjectionMatrix().m, (ImGuizmo::OPERATION)m_GizmoOperation, ImGuizmo::MODE::LOCAL, (float*)transformMatrix.m, 0, snapping))
					{
						Vector3 scale;
						Quaternion rotation;
						Vector3 translation;

						if (parentTransform != nullptr)
						{
							transformMatrix *= parentTransform->GetLocalToWorldMatrix().Invert();
						}

						transformMatrix.Decompose(scale, rotation, translation);
						transform->SetLocalPosition(translation);
						transform->SetLocalRotation(rotation);
						transform->SetLocalScale(scale);
					}
				}
			}
		}
	}

	void SceneArea::DrawScene(const float width, const float height)
	{
		GfxDevice::SetRenderTarget(m_SceneRenderTarget, m_SceneDepthStencil);
		GfxDevice::SetViewport(0, 0, static_cast<int>(width), static_cast<int>(height));
		GfxDevice::ClearColor({ 0.117f, 0.117f, 0.117f, 1 });
		GfxDevice::ClearDepth(1.0f);

		Scene* scene = EditorSceneManager::GetScene();
		if (scene != nullptr)
		{
			SceneRenderer::Draw(scene, &m_Camera);
		}
		
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