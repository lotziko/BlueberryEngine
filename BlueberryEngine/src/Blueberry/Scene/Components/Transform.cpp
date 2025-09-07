#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Time.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Transform, Component)
	{
		DEFINE_BASE_FIELDS(Transform, Component)
		DEFINE_FIELD(Transform, m_LocalPosition, BindingType::Vector3, FieldOptions().SetUpdateCallback(MethodBind::Create(&Transform::InvalidateHierarchy)))
		DEFINE_FIELD(Transform, m_LocalRotation, BindingType::Quaternion, FieldOptions().SetUpdateCallback(MethodBind::Create(&Transform::InvalidateHierarchy)))
		DEFINE_FIELD(Transform, m_LocalScale, BindingType::Vector3, FieldOptions().SetUpdateCallback(MethodBind::Create(&Transform::InvalidateHierarchy)))
		DEFINE_FIELD(Transform, m_LocalRotationEulerHint, BindingType::Vector3, FieldOptions().SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(Transform, m_Parent, BindingType::ObjectPtr, FieldOptions().SetObjectType(Transform::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(Transform, m_Children, BindingType::ObjectPtrList, FieldOptions().SetObjectType(Transform::Type).SetVisibility(VisibilityType::Hidden))
	}

	void Transform::OnEnable()
	{
		if (m_IsDirty)
		{
			RecalculateHierarchy();
		}
	}

	void Transform::OnDestroy()
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
	}

	const Matrix& Transform::GetLocalToWorldMatrix()
	{
		if (m_IsDirty)
		{
			RecalculateHierarchy();
		}

		return m_LocalToWorldMatrix;
	}

	const Matrix& Transform::GetWorldToLocalMatrix()
	{
		if (m_IsDirty)
		{
			RecalculateHierarchy();
		}

		return m_WorldToLocalMatrix;
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

	const Vector3 Transform::GetLocalEulerRotationHint() const
	{
		return m_LocalRotationEulerHint;
	}

	const Vector3 Transform::GetPosition()
	{
		if (m_IsDirty)
		{
			RecalculateHierarchy();
		}

		return m_Position;
	}

	const Quaternion Transform::GetRotation()
	{
		if (m_IsDirty)
		{
			RecalculateHierarchy();
		}

		return m_Rotation;
	}

	Transform* Transform::GetParent() const
	{
		return m_Parent.Get();
	}

	const List<Transform*> Transform::GetChildren() const
	{
		List<Transform*> children;
		for (auto child : m_Children)
		{
			if (child.IsValid())
			{
				children.emplace_back(child.Get());
			}
		}
		return children;
	}

	const size_t Transform::GetChildrenCount() const
	{
		return m_Children.size();
	}

	void Transform::SetLocalPosition(const Vector3& position)
	{
		m_LocalPosition = position;
		InvalidateHierarchy();
	}

	void Transform::SetLocalRotation(const Quaternion& rotation)
	{
		m_LocalRotation = rotation;
		InvalidateHierarchy();
	}

	void Transform::SetLocalEulerRotation(const Vector3& euler)
	{
		m_LocalRotation = Quaternion::CreateFromYawPitchRoll(euler.y, euler.x, euler.z);
		InvalidateHierarchy();
	}

	void Transform::SetLocalEulerRotationHint(const Vector3& euler)
	{
		m_LocalRotationEulerHint = euler;
		m_LocalRotation = Quaternion::CreateFromYawPitchRoll(ToRadians(m_LocalRotationEulerHint.y), ToRadians(m_LocalRotationEulerHint.x), ToRadians(m_LocalRotationEulerHint.z));
		m_LocalRotation.Normalize();
		InvalidateHierarchy();
	}

	void Transform::SetLocalScale(const Vector3& scale)
	{
		m_LocalScale = scale;
		InvalidateHierarchy();
	}

	void Transform::SetPosition(const Vector3& position)
	{
		if (!m_Parent.IsValid())
		{
			m_LocalPosition = position;
		}
		else
		{
			m_LocalPosition = Vector3::Transform(position, m_Parent->m_WorldToLocalMatrix);
		}
		InvalidateHierarchy();
	}

	void Transform::SetRotation(const Quaternion& rotation)
	{
		if (!m_Parent.IsValid())
		{
			m_LocalRotation = rotation;
		}
		else
		{
			m_LocalRotation = Quaternion::CreateFromRotationMatrix(Matrix::Transform(m_Parent->m_WorldToLocalMatrix, rotation));
		}
		InvalidateHierarchy();
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
		// Reset activity
		m_Entity->UpdateHierarchy(true);
	}

	const bool& Transform::IsDirty() const
	{
		return m_IsDirty;
	}

	const size_t& Transform::GetRecalculationFrame() const
	{
		return m_RecalculationFrame;
	}

	void Transform::InvalidateHierarchy()
	{
		m_IsDirty = true;
		for (auto& child : m_Children)
		{
			child->InvalidateHierarchy();
		}
	}

	void Transform::RecalculateHierarchy()
	{
		m_LocalMatrix = CreateTRS(m_LocalPosition, m_LocalRotation, m_LocalScale);
		m_IsDirty = false;
		if (m_Parent.IsValid())
		{
			m_Parent.Get()->RecalculateHierarchy();
			m_LocalToWorldMatrix = m_LocalMatrix * (m_Parent->m_LocalToWorldMatrix);
			m_LocalToWorldMatrix.Decompose(m_Scale, m_Rotation, m_Position);
		}
		else
		{
			m_LocalToWorldMatrix = m_LocalMatrix;
			m_Position = m_LocalPosition;
			m_Rotation = m_LocalRotation;
			m_Scale = m_LocalScale;
		}
		m_LocalToWorldMatrix.Invert(m_WorldToLocalMatrix);
		m_RecalculationFrame = Time::GetFrameCount();
	}
}