#pragma once

#include "Component.h"

class Transform : public Component
{
	OBJECT_DECLARATION(Transform)

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

	void SetLocalPosition(const Vector3& position)
	{
		m_LocalPosition = position;
		m_IsDirty = true;
	}

	void SetLocalRotation(const Vector3& euler)
	{
		m_LocalRotation = Quaternion::CreateFromYawPitchRoll(euler.x, euler.y, euler.z);
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