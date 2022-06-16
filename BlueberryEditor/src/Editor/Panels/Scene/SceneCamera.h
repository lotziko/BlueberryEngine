#pragma once

#include "Blueberry\Math\Math.h"

namespace Blueberry
{
	class SceneCamera
	{
	public:
		void Update();

		const Matrix& GetProjectionMatrix() const { return m_ProjectionMatrix; }
		const Matrix& GetViewMatrix() const { return m_ViewMatrix; }

		void SetViewport(const Viewport& viewport) { m_Viewport = viewport; }

	private:
		Viewport m_Viewport;
		Matrix m_ProjectionMatrix;
		Matrix m_ViewMatrix;

		Vector3 m_Direction = Vector3(0, 0, -1);
		Vector3 m_Up = Vector3(0, 1, 0);

		float m_Zoom = 0.025f;
	};
}