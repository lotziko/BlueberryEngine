#pragma once

#include "Component.h"

namespace Blueberry
{
	class Transform : public Component
	{
		OBJECT_DECLARATION(Transform)

	public:
		Transform();
		~Transform();
		
		const Matrix& GetLocalToWorldMatrix();
		const Vector3& GetLocalPosition();
		const Quaternion& GetLocalRotation();
		const Vector3& GetLocalScale();
		const Vector3 GetLocalEulerRotation() const;

		const Ref<Transform>& GetParent() const;

		const std::vector<Ref<Transform>>& GetChildren() const;
		const std::size_t GetChildrenCount() const;

		void SetLocalPosition(const Vector3& position);
		void SetLocalRotation(const Quaternion& rotation);
		void SetLocalEulerRotation(const Vector3& euler);
		void SetLocalScale(const Vector3& scale);

		void SetParent(Ref<Transform>& parent);

		void Update();

		const bool& IsDirty() const;

		static void BindProperties();

	private:
		void RecalculateWorldMatrix(bool dirty);

	private:
		bool m_IsDirty = true;

		Ref<Transform> m_Parent;
		std::vector<Ref<Transform>> m_Children;

		Matrix m_LocalToWorldMatrix;
		Matrix m_LocalMatrix;

		Vector3 m_LocalPosition;
		Quaternion m_LocalRotation;
		Vector3 m_LocalScale;
	};
}