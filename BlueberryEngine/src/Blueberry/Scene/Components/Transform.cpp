#include "bbpch.h"
#include "Transform.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Transform)

	void Transform::Destroy()
	{
		if (m_Parent.IsValid())
		{
			if (m_Parent->m_Children.size() > 0)
			{
				auto index = std::find_if(m_Parent->m_Children.begin(), m_Parent->m_Children.end(), [this](ObjectPtr<Transform> const& i) { return i.Get() == this; });
				m_Parent->m_Children.erase(index);
				m_Parent = nullptr;
			}
		}
		for (auto child : m_Children)
		{
			if (child.IsValid())
			{
				child->GetEntity()->GetScene()->DestroyEntity(child->GetEntity());
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

	Transform* Transform::GetParent() const
	{
		return m_Parent.Get();
	}

	const std::vector<Transform*> Transform::GetChildren() const
	{
		std::vector<Transform*> children;
		for (auto child : m_Children)
		{
			if (child.IsValid())
			{
				children.emplace_back(child.Get());
			}
		}
		return children;
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
		// TODO worldTransformStays
		if (m_Parent.IsValid())
		{
			if (m_Parent->m_Children.size() > 0)
			{
				auto index = std::find_if(m_Parent->m_Children.begin(), m_Parent->m_Children.end(), [this](ObjectPtr<Transform> const& i) { return i.Get() == this; });
				m_Parent->m_Children.erase(index);
			}
		}
		m_Parent = parent;
		m_Parent->m_Children.emplace_back(this);
	}

	void Transform::Update()
	{
		if (!m_Parent.IsValid())
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
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &Transform::m_Entity, BindingType::ObjectPtr, Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_LocalPosition), &Transform::m_LocalPosition, BindingType::Vector3))
		BIND_FIELD(FieldInfo(TO_STRING(m_LocalRotation), &Transform::m_LocalRotation, BindingType::Quaternion))
		BIND_FIELD(FieldInfo(TO_STRING(m_LocalScale), &Transform::m_LocalScale, BindingType::Vector3))
		BIND_FIELD(FieldInfo(TO_STRING(m_Parent), &Transform::m_Parent, BindingType::ObjectPtr, Transform::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Children), &Transform::m_Children, BindingType::ObjectPtrArray, Transform::Type))
		END_OBJECT_BINDING()
	}

	void Transform::RecalculateWorldMatrix(bool dirty)
	{
		dirty |= m_IsDirty;

		if (dirty)
		{
			m_LocalMatrix = Matrix::CreateScale(m_LocalScale) * Matrix::CreateFromQuaternion(m_LocalRotation) * Matrix::CreateTranslation(m_LocalPosition);
			m_LocalToWorldMatrix = !m_Parent.IsValid() ? m_LocalMatrix : (m_LocalMatrix * (m_Parent->m_LocalToWorldMatrix));
			m_IsDirty = false;
		}

		for (auto child : m_Children)
		{
			child->RecalculateWorldMatrix(dirty);
		}
	}

}