#include "bbpch.h"
#include "Collider.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\PhysicsBody.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Component, Collider)

	void Collider::OnBeginPlay()
	{
		Transform* transform = GetTransform();
		while (transform != nullptr)
		{
			PhysicsBody* body = transform->GetEntity()->GetComponent<PhysicsBody>();
			if (body != nullptr)
			{
				body->m_Colliders.emplace_back(this);
				break;
			}
			transform = transform->GetParent();
		}
	}
}