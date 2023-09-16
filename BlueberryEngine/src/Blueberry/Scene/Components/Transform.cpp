#include "bbpch.h"
#include "Transform.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Transform)

	Transform::Transform()
	{
		m_LocalScale = Vector3(1, 1, 1);
		m_Parent = nullptr;
	}

	Transform::~Transform()
	{
		if (m_Parent != nullptr)
		{
			if (m_Parent->m_Children.size() > 0)
			{
				auto index = std::find(m_Parent->m_Children.begin(), m_Parent->m_Children.end(), this);
				m_Parent->m_Children.erase(index);
				m_Parent = nullptr;
			}
		}
	}

	const Matrix& Transform::GetLocalToWorldMatrix()
	{
		if (m_IsDirty)
		{
			//RecalculateWorldMatrix();
			m_IsDirty = false;
		}

		return m_LocalToWorldMatrix;
	}

	const Vector3& Transform::GetLocalPosition() const
	{
		return m_LocalPosition;
	}

	const Quaternion& Transform::GetLocalRotation() const
	{
		return m_LocalRotation;
	}

	const Vector3& Transform::GetLocalScale() const
	{
		return m_LocalScale;
	}

	const Vector3 Transform::GetLocalEulerRotation() const
	{
		return m_LocalRotation.ToEuler();
	}

	Transform* Transform::GetParent()
	{
		return m_Parent;
	}

	const std::vector<Transform*>& Transform::GetChildren() const
	{
		return m_Children;
	}

	const std::size_t Transform::GetChildrenCount() const
	{
		return m_Children.size();
	}

	void Transform::SetLocalPosition(const Vector3& position)
	{
		m_LocalPosition = position;
		m_IsDirty = true;
	}

	void Transform::SetLocalRotation(const Quaternion& rotation)
	{
		m_LocalRotation = rotation;
		m_IsDirty = true;
	}

	void Transform::SetLocalEulerRotation(const Vector3& euler)
	{
		m_LocalRotation = Quaternion::CreateFromYawPitchRoll(euler.x, euler.y, euler.z);
		m_IsDirty = true;
	}

	void Transform::SetLocalScale(const Vector3& scale)
	{
		m_LocalScale = scale;
		m_IsDirty = true;
	}

	void Transform::SetParent(Transform* parent)
	{
		m_Parent = parent;
		m_Parent->m_Children.emplace_back(this);
	}

	void Transform::Update()
	{
		if (m_Parent == nullptr)
		{
			RecalculateWorldMatrix(m_IsDirty);
		}
	}

	const bool& Transform::IsDirty() const
	{
		return m_IsDirty;
	}

	std::string Transform::ToString() const
	{
		return "Transform";
	}

	void Transform::RecalculateWorldMatrix(bool dirty)
	{
		dirty |= m_IsDirty;

		if (dirty)
		{
			m_LocalMatrix = Matrix::CreateScale(m_LocalScale) * Matrix::CreateFromQuaternion(m_LocalRotation) * Matrix::CreateTranslation(m_LocalPosition);
			m_LocalToWorldMatrix = m_Parent == nullptr ? m_LocalMatrix : (m_LocalMatrix * (m_Parent->m_LocalToWorldMatrix));
			m_IsDirty = false;
		}

		for (auto child : m_Children)
		{
			child->RecalculateWorldMatrix(dirty);
		}
	}

}