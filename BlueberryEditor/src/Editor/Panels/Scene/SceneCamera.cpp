#include "bbpch.h"
#include "SceneCamera.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	void SceneCamera::Update()
	{
		ImGuiIO *io = &ImGui::GetIO();

		float mouseWheel = io->MouseWheel;
		if (mouseWheel != 0)
		{
			m_Zoom += mouseWheel / 25;
		}

		if (ImGui::IsMouseDragging(1))
		{
			if (m_IsDragging == false)
			{
				ImVec2 pos = ImGui::GetCursorScreenPos();
				ImVec2 size = ImGui::GetContentRegionAvail();
				ImVec2 clickPos = io->MouseClickedPos[1];
				if (clickPos.x >= pos.x && clickPos.y >= pos.y && clickPos.x <= pos.x + size.x && clickPos.y <= pos.y + size.y)
				{
					m_IsDragging = true;
				}
			}
			else
			{
				ImVec2 dragDelta = ImGui::GetMouseDragDelta(1);
				m_DragDeltaPosition.x = -dragDelta.x / m_Zoom;
				m_DragDeltaPosition.y = dragDelta.y / m_Zoom;
			}
		}
		else
		{
			if (m_IsDragging)
			{
				m_IsDragging = false;
				m_Position += m_DragDeltaPosition;
				m_DragDeltaPosition = Vector3::Zero;
			}
		}
	}

	void SceneCamera::UpdateMatrices()
	{
		Matrix rotationMatrix = Matrix::Identity;

		Vector3 position = m_Position + m_DragDeltaPosition;
		Vector3 target = Vector3::Transform(m_Direction, rotationMatrix);
		target += position;

		Vector3 up = Vector3::Transform(m_Up, rotationMatrix);

		m_ViewMatrix = Matrix::CreateLookAt(position, target, up);
		m_ProjectionMatrix = Matrix::CreateOrthographic(m_Viewport.width / m_Zoom, m_Viewport.height / m_Zoom, 0.1f, 1000.0f);
	}
}