#include "Blueberry\Scene\Components\Collider.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\PhysicsBody.h"
#include "Blueberry\Physics\Physics.h"

#include "Blueberry\Core\ClassDB.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\Body\BodyID.h>
#include <Jolt\Physics\PhysicsSystem.h>
#include <Jolt\Physics\Body\BodyCreationSettings.h>

namespace Blueberry
{
	OBJECT_DEFINITION(Collider, Component)
	{
		DEFINE_BASE_FIELDS(Collider, Component)
	}

	struct Collider::PrivateData
	{
		JPH::BodyID bodyId;
	};

	Collider::Collider()
	{
		ZeroMemory(m_PrivateStorage, sizeof(PrivateData));
		m_PrivateData = reinterpret_cast<PrivateData*>(&m_PrivateStorage);
	}

	void Collider::OnCreate()
	{
		PhysicsBody* body = nullptr;
		Transform* transform = GetTransform();
		while (transform != nullptr)
		{
			body = transform->GetEntity()->GetComponent<PhysicsBody>();
			if (body != nullptr)
			{
				break;
			}
			transform = transform->GetParent();
		}
		if (body == nullptr)
		{
			Transform* transform = GetTransform();
			JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
			JPH::Shape* shape = GetShape();
			if (shape != nullptr)
			{
				Vector3 position = transform->GetPosition();
				Quaternion rotation = transform->GetRotation();
				Vector3 scale = transform->GetScale();
				if (scale.x * scale.y * scale.z != 1.0f)
				{
					JPH::Result result = shape->ScaleShape(JPH::RVec3(std::abs(scale.x), std::abs(scale.y), std::abs(scale.z)));
					JPH::BodyCreationSettings settings(result.Get().GetPtr(), JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), JPH::EMotionType::Static, 1);
					m_PrivateData->bodyId = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
				}
				else
				{
					JPH::BodyCreationSettings settings(shape, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), JPH::EMotionType::Static, 1);
					m_PrivateData->bodyId = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
				}
			}
		}
		else
		{
			body->m_Colliders.push_back(this);
		}
	}

	void Collider::OnDestroy()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.DestroyBody(m_PrivateData->bodyId);
		}
	}

	void Collider::OnEnable()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (!bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.AddBody(m_PrivateData->bodyId, JPH::EActivation::Activate);
		}
	}

	void Collider::OnDisable()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.RemoveBody(m_PrivateData->bodyId);
		}
	}
}