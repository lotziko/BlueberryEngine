#pragma once

#include "Component.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Transform : public Component
	{
		OBJECT_DECLARATION(Transform)

	public:
		Transform() = default;
		~Transform() = default;

		virtual void OnCreate() override;
		virtual void OnDestroy() override;
		
		const Matrix& GetLocalToWorldMatrix();
		const Vector3& GetLocalPosition();
		const Quaternion& GetLocalRotation();
		const Vector3& GetLocalScale();
		const Vector3 GetLocalEulerRotation() const;
		const Vector3 GetPosition();
		const Quaternion GetRotation();

		Transform* GetParent() const;

		const std::vector<Transform*> GetChildren() const;
		const std::size_t GetChildrenCount() const;

		void SetLocalPosition(const Vector3& position);
		void SetLocalRotation(const Quaternion& rotation);
		void SetLocalEulerRotation(const Vector3& euler);
		void SetLocalScale(const Vector3& scale);

		void SetPosition(const Vector3& position);
		void SetRotation(const Quaternion& rotation);

		void SetParent(Transform* parent);

		const bool& IsDirty() const;
		const size_t& GetRecalculationFrame() const;

		static void BindProperties();

	private:
		void SetHierarchyDirty();
		void RecalculateHierarchy();

	private:
		bool m_IsDirty = true;
		size_t m_RecalculationFrame = 0;

		ObjectPtr<Transform> m_Parent = nullptr;
		std::vector<ObjectPtr<Transform>> m_Children;

		Matrix m_LocalToWorldMatrix;
		Matrix m_WorldToLocalMatrix;
		Matrix m_LocalMatrix;

		Vector3 m_LocalPosition;
		Quaternion m_LocalRotation;
		Vector3 m_LocalScale = Vector3(1, 1, 1);

		Vector3 m_Position;
		Quaternion m_Rotation;
		Vector3 m_Scale;
	};
}