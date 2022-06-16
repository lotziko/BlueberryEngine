#pragma once

#include "Component.h"

namespace Blueberry
{
	class Transform : public Component
	{
		OBJECT_DECLARATION(Transform)
		COMPONENT_DECLARATION(Transform)

	public:
		Transform()
		{
			m_LocalScale = Vector3(1, 1, 1);
		}
		~Transform() = default;

		const Matrix& GetWorldMatrix()
		{
			if (m_IsDirty)
			{
				RecalculateWorldMatrix();
				m_IsDirty = false;
			}

			return m_WorldMatrix;
		}
		const Vector3& GetLocalPosition() const { return m_LocalPosition; }
		const Quaternion& GetLocalRotation() const { return m_LocalRotation; }
		const Vector3& GetLocalScale() const { return m_LocalScale; }

		Vector3 GetLocalEulerRotation() const { return m_LocalRotation.ToEuler(); }

		void SetLocalPosition(const Vector3& position)
		{
			m_LocalPosition = position;
			m_IsDirty = true;
		}

		void SetLocalRotation(const Quaternion& rotation)
		{
			m_LocalRotation = rotation;
			m_IsDirty = true;
		}

		void SetLocalEulerRotation(const Vector3& euler)
		{
			m_LocalRotation = Quaternion::CreateFromYawPitchRoll(euler.x, euler.y, euler.z);
			m_IsDirty = true;
		}

		void SetLocalScale(const Vector3& scale)
		{
			m_LocalScale = scale;
			m_IsDirty = true;
		}

		const bool& IsDirty() const { return m_IsDirty; }

	private:
		void RecalculateWorldMatrix()
		{
			m_WorldMatrix = Matrix::CreateTranslation(m_LocalPosition);
		}

	private:
		bool m_IsDirty = true;

		Transform* m_Parent;
		std::vector<Transform*> m_Children;

		Matrix m_WorldMatrix;
		Matrix m_LocalMatrix;

		Vector3 m_LocalPosition;
		Quaternion m_LocalRotation;
		Vector3 m_LocalScale;
	};
}