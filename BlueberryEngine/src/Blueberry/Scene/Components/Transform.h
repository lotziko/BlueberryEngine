#pragma once

#include "Component.h"

namespace Blueberry
{
	class Transform : public Component
	{
		OBJECT_DECLARATION(Transform)
		COMPONENT_DECLARATION(Transform)

	public:
		Transform();
		~Transform();

		const Matrix& GetLocalToWorldMatrix();
		const Vector3& GetLocalPosition() const;
		const Quaternion& GetLocalRotation() const;
		const Vector3& GetLocalScale() const;
		const Vector3 GetLocalEulerRotation() const;

		Transform* GetParent();

		const std::vector<Transform*>& GetChildren() const;
		const std::size_t GetChildrenCount() const;

		void SetLocalPosition(const Vector3& position);
		void SetLocalRotation(const Quaternion& rotation);
		void SetLocalEulerRotation(const Vector3& euler);
		void SetLocalScale(const Vector3& scale);

		void SetParent(Transform* parent);

		void Update();

		const bool& IsDirty() const;

		virtual std::string ToString() const final;

	private:
		void RecalculateWorldMatrix(bool dirty);

	private:
		bool m_IsDirty = true;

		Transform* m_Parent;
		std::vector<Transform*> m_Children;

		Matrix m_LocalToWorldMatrix;
		Matrix m_LocalMatrix;

		Vector3 m_LocalPosition;
		Quaternion m_LocalRotation;
		Vector3 m_LocalScale;
	};
}