#include "bbpch.h"
#include "SceneCamera.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	void SceneCamera::Update()
	{
		Matrix rotationMatrix = Matrix::Identity;

		Vector3 position = Vector3::Zero;
		Vector3 target = Vector3::Transform(m_Direction, rotationMatrix);
		target += position;

		Vector3 up = Vector3::Transform(m_Up, rotationMatrix);

		m_ViewMatrix = Matrix::CreateLookAt(position, target, up);
		m_ProjectionMatrix = Matrix::CreateOrthographic(m_Viewport.width * m_Zoom, m_Viewport.height * m_Zoom, 0.1f, 1000.0f);
	}
}