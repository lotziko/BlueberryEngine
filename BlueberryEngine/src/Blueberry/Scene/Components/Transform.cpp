#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	static const uint8_t DIRTY_LOCAL = 1;
	static const uint8_t DIRTY_WORLD = 2;
	static const uint8_t DIRTY_WORLD_TO_LOCAL = 4;
	static const uint8_t DIRTY_ALL = 7;

	OBJECT_DEFINITION(Transform, Component)
	{
		DEFINE_BASE_FIELDS(Transform, Component)
		DEFINE_FIELD(Transform, m_LocalPosition, BindingType::Vector3, FieldOptions().SetUpdateCallback(MethodBind::Create(&Transform::InvalidateHierarchy)))
		DEFINE_FIELD(Transform, m_LocalRotation, BindingType::Quaternion, FieldOptions().SetUpdateCallback(MethodBind::Create(&Transform::InvalidateHierarchy)))
		DEFINE_FIELD(Transform, m_LocalScale, BindingType::Vector3, FieldOptions().SetUpdateCallback(MethodBind::Create(&Transform::InvalidateHierarchy)))
		DEFINE_FIELD(Transform, m_IsStatic, BindingType::Bool, FieldOptions().SetUpdateCallback(MethodBind::Create(&Transform::InvalidateHierarchy)))
		DEFINE_FIELD(Transform, m_LocalRotationEulerHint, BindingType::Vector3, FieldOptions().SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(Transform, m_Parent, BindingType::ObjectPtr, FieldOptions().SetObjectType(Transform::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(Transform, m_Children, BindingType::ObjectPtrList, FieldOptions().SetObjectType(Transform::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_EXECUTE_ALWAYS()
	}

	void Transform::OnEnable()
	{
		if (m_DirtyFlags)
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
		if (m_DirtyFlags & DIRTY_WORLD)
		{
			RecalculateHierarchy();
		}

		return m_LocalToWorldMatrix;
	}

	const Matrix& Transform::GetWorldToLocalMatrix()
	{
		if (m_DirtyFlags & DIRTY_WORLD)
		{
			RecalculateHierarchy();
		}

		if (m_DirtyFlags & DIRTY_WORLD_TO_LOCAL)
		{
			m_LocalToWorldMatrix.Invert(m_WorldToLocalMatrix);
			m_DirtyFlags &= ~DIRTY_WORLD_TO_LOCAL;
		}

		return m_WorldToLocalMatrix;
	}

	const Matrix& Transform::GetLocalMatrix()
	{
		if (m_DirtyFlags & DIRTY_LOCAL)
		{
			m_LocalMatrix = Math::CreateTRS(m_LocalPosition, m_LocalRotation, m_LocalScale);
			m_DirtyFlags &= ~DIRTY_LOCAL;
			++m_UpdateCount;
		}

		return m_LocalMatrix;
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
		if (m_DirtyFlags & DIRTY_WORLD)
		{
			RecalculateHierarchy();
		}

		return m_Position;
	}

	const Quaternion Transform::GetRotation()
	{
		if (m_DirtyFlags & DIRTY_WORLD)
		{
			RecalculateHierarchy();
		}

		return m_Rotation;
	}

	const Vector3 Transform::GetScale()
	{
		if (m_DirtyFlags & DIRTY_WORLD)
		{
			RecalculateHierarchy();
		}

		return m_Scale;
	}

	Transform* Transform::GetParent() const
	{
		return m_Parent.Get();
	}

	const List<ObjectPtr<Transform>>& Transform::GetChildren() const
	{
		return m_Children;
	}

	Transform* Transform::GetChild(const size_t& index)
	{
		return m_Children[index].Get();
	}

	const size_t Transform::GetChildrenCount() const
	{
		return m_Children.size();
	}

	const size_t Transform::GetSiblingIndex()
	{
		if (m_Parent.IsValid())
		{
			auto& children = m_Parent->m_Children;
			for (size_t i = 0; i < children.size(); ++i)
			{
				if (children[i]->GetObjectId() == m_ObjectId)
				{
					return i;
				}
			}
		}
		else
		{
			return GetScene()->GetRootIndex(m_Entity.Get());
		}
		return 0;
	}

	void Transform::SetLocalPosition(const Vector3& position)
	{
		m_LocalPosition = position;
		InvalidateHierarchy();
	}

	void Transform::SetLocalRotation(const Quaternion& rotation)
	{
		m_LocalRotation = rotation;
		m_LocalRotationEulerHint = Math::ToDegrees(m_LocalRotation.ToEuler());
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
		m_LocalRotation = Quaternion::CreateFromYawPitchRoll(Math::ToRadians(m_LocalRotationEulerHint.y), Math::ToRadians(m_LocalRotationEulerHint.x), Math::ToRadians(m_LocalRotationEulerHint.z));
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
			m_LocalPosition = Vector3::Transform(position, m_Parent->GetWorldToLocalMatrix());
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
			m_LocalRotation = Quaternion::CreateFromRotationMatrix(Matrix::Transform(m_Parent->GetWorldToLocalMatrix(), rotation));
		}
		m_LocalRotationEulerHint = Math::ToDegrees(m_LocalRotation.ToEuler());
		InvalidateHierarchy();
	}

	void Transform::SetLocalTRS(const TRS& trs)
	{
		m_LocalPosition = trs.position;
		m_LocalRotation = trs.rotation;
		m_LocalScale = trs.scale;
		InvalidateHierarchy();
	}

	void Transform::SetParent(Transform* parent, const bool& worldPositionStays)
	{
		if (m_Parent == parent)
		{
			return;
		}

		Scene* scene = GetScene();
		if (scene != nullptr && (parent != nullptr) != m_Parent.IsValid())
		{
			if (parent == nullptr)
			{
				scene->AddToRoot(GetEntity());
			}
			else
			{
				scene->RemoveFromRoot(GetEntity());
			}
		}

		Vector3 previousPosition;
		Quaternion previousRotation;
		if (worldPositionStays)
		{
			previousPosition = GetPosition();
			previousRotation = GetRotation();
		}

		if (m_Parent.IsValid())
		{
			if (m_Parent->m_Children.size() > 0)
			{
				m_Parent->m_Children.erase(std::find_if(m_Parent->m_Children.begin(), m_Parent->m_Children.end(), [this](ObjectPtr<Transform> const& i) { return i.Get() == this; }));
			}
		}
		m_Parent = parent;
		if (parent != nullptr)
		{
			m_Parent->m_Children.push_back(this);
		}

		if (worldPositionStays)
		{
			SetPosition(previousPosition);
			SetRotation(previousRotation);
		}
		else
		{
			InvalidateHierarchy();
		}

		m_Entity->UpdateHierarchy();
	}

	void Transform::SetSiblingIndex(const size_t& index)
	{
		if (m_Parent.IsValid())
		{
			size_t oldIndex = 0;
			auto& children = m_Parent->m_Children;
			for (size_t i = 0; i < children.size(); ++i)
			{
				if (children[i].Get() == this)
				{
					oldIndex = i;
					break;
				}
			}
			children.move_element(oldIndex, index);
		}
		else
		{
			GetScene()->SetRootIndex(m_Entity.Get(), index);
		}
	}

	const size_t& Transform::GetUpdateCount()
	{
		if (m_DirtyFlags & (DIRTY_LOCAL | DIRTY_WORLD))
		{
			RecalculateHierarchy();
		}

		return m_UpdateCount;
	}

	const bool& Transform::IsStatic() const
	{
		return m_IsStatic;
	}

	void Transform::SetStatic(const bool& isStatic)
	{
		m_IsStatic = isStatic;
		InvalidateHierarchy();
	}

	void Transform::InvalidateHierarchy()
	{
		if (m_DirtyFlags ^ DIRTY_ALL)
		{
			m_DirtyFlags = DIRTY_ALL;
			for (auto& child : m_Children)
			{
				child->InvalidateHierarchy();
			}
		}
	}

	void Transform::RecalculateHierarchy()
	{
		if (m_DirtyFlags & DIRTY_LOCAL)
		{
			m_LocalMatrix = Math::CreateTRS(m_LocalPosition, m_LocalRotation, m_LocalScale);
			m_DirtyFlags &= ~DIRTY_LOCAL;
		}
		if (m_DirtyFlags & DIRTY_WORLD)
		{
			if (m_Parent.IsValid())
			{
				if (m_Parent->m_DirtyFlags)
				{
					m_Parent.Get()->RecalculateHierarchy();
				}
				m_LocalToWorldMatrix = m_LocalMatrix * (m_Parent->m_LocalToWorldMatrix);
				m_Position = m_Parent->m_Position + Vector3::Transform(m_Parent->m_Scale * m_LocalPosition, m_Parent->m_Rotation);
				m_Rotation = m_LocalRotation * m_Parent->m_Rotation;
				m_Scale = m_Parent->m_Scale * m_LocalScale;
			}
			else
			{
				m_LocalToWorldMatrix = m_LocalMatrix;
				m_Position = m_LocalPosition;
				m_Rotation = m_LocalRotation;
				m_Scale = m_LocalScale;
			}
			m_DirtyFlags &= ~DIRTY_WORLD;
		}
		++m_UpdateCount;
	}
}