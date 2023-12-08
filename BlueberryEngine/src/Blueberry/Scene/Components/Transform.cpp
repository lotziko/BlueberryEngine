#include "bbpch.h"
#include "Transform.h"

#include "Blueberry\Core\ClassDB.h"

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
				auto index = std::find_if(m_Parent->m_Children.begin(), m_Parent->m_Children.end(), [this](std::shared_ptr<Transform> const& i) { return i.get() == this; });
				m_Parent->m_Children.erase(index);
				m_Parent = nullptr;
			}
		}
	}

	const Matrix& Transform::GetLocalToWorldMatrix()
	{
		if (m_IsDirty)
		{
			RecalculateWorldMatrix(true);
			m_IsDirty = false;
		}

		return m_LocalToWorldMatrix;
	}

	const Vector3& Transform::GetLocalPosition()
	{
		return m_LocalPosition;
	}

	const Quaternion& Transform::GetLocalRotation()
	{
		return m_LocalRotation;
	}

	const Vector3& Transform::GetLocalScale()
	{
		return m_LocalScale;
	}

	const Vector3 Transform::GetLocalEulerRotation() const
	{
		return m_LocalRotation.ToEuler();
	}

	const Ref<Transform>& Transform::GetParent() const
	{
		return m_Parent;
	}

	const std::vector<Ref<Transform>>& Transform::GetChildren() const
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

	void Transform::SetParent(Ref<Transform>& parent)
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

	void Transform::BindProperties()
	{
		BEGIN_OBJECT_BINDING(Transform)
		BIND_FIELD("m_Entity", &Transform::m_Entity, BindingType::Object)
		BIND_FIELD("m_LocalPosition", &Transform::m_LocalPosition, BindingType::Vector3)
		BIND_FIELD("m_LocalRotation", &Transform::m_LocalRotation, BindingType::Quaternion)
		BIND_FIELD("m_LocalScale", &Transform::m_LocalScale, BindingType::Vector3)
		BIND_FIELD("m_Parent", &Transform::m_Parent, BindingType::ObjectRef)
		BIND_FIELD("m_Children", &Transform::m_Children, BindingType::ObjectRefArray)
		END_OBJECT_BINDING()
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