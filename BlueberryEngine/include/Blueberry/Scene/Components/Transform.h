#pragma once

#include "Component.h"
#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class BB_API Transform : public Component
	{
		OBJECT_DECLARATION(Transform)

	public:
		Transform() = default;
		~Transform() = default;

		virtual void OnEnable() override;
		virtual void OnDestroy() override;
		
		const Matrix& GetLocalToWorldMatrix();
		const Matrix& GetWorldToLocalMatrix();
		const Matrix& GetLocalMatrix();
		const Vector3& GetLocalPosition();
		const Quaternion& GetLocalRotation();
		const Vector3& GetLocalScale();
		const Vector3 GetLocalEulerRotation() const;
		const Vector3 GetLocalEulerRotationHint() const;
		const Vector3 GetPosition();
		const Quaternion GetRotation();

		Transform* GetParent() const;

		const List<ObjectPtr<Transform>>& GetChildren() const;
		Transform* GetChild(const size_t& index);
		const size_t GetChildrenCount() const;

		const size_t GetSiblingIndex();

		void SetLocalPosition(const Vector3& position);
		void SetLocalRotation(const Quaternion& rotation);
		void SetLocalEulerRotation(const Vector3& euler);
		void SetLocalEulerRotationHint(const Vector3& euler);
		void SetLocalScale(const Vector3& scale);

		void SetPosition(const Vector3& position);
		void SetRotation(const Quaternion& rotation);
		void SetLocalTRS(const TRS& trs);

		void SetParent(Transform* parent, const bool& worldPositionStays = true);
		void SetSiblingIndex(const size_t& index);

		const size_t& GetUpdateCount();

		const bool& IsStatic() const;
		void SetStatic(const bool& isStatic);

	private:
		void InvalidateHierarchy();
		void RecalculateHierarchy();

	private:
		uint8_t m_DirtyFlags = 7;
		size_t m_UpdateCount = 0;

		ObjectPtr<Transform> m_Parent = nullptr;
		List<ObjectPtr<Transform>> m_Children;

		Matrix m_LocalToWorldMatrix;
		Matrix m_WorldToLocalMatrix;
		Matrix m_LocalMatrix;

		Vector3 m_LocalPosition;
		Quaternion m_LocalRotation;
		Vector3 m_LocalScale = Vector3(1, 1, 1);
		Vector3 m_LocalRotationEulerHint;
		bool m_IsStatic = true;

		Vector3 m_Position;
		Quaternion m_Rotation;
		Vector3 m_Scale;
	};
}