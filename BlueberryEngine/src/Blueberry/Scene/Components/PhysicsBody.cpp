#include "Blueberry\Scene\Components\PhysicsBody.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Collider.h"
#include "Blueberry\Physics\Physics.h"
#include "Blueberry\Core\ClassDB.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\Body\BodyID.h>
#include <Jolt\Physics\PhysicsSystem.h>
#include <Jolt\Physics\Collision\Shape\StaticCompoundShape.h>
#include <Jolt\Physics\Body\BodyCreationSettings.h>

namespace Blueberry
{
	OBJECT_DEFINITION(PhysicsBody, Component)
	{
		DEFINE_BASE_FIELDS(PhysicsBody, Component)
		DEFINE_FIELD(PhysicsBody, m_IsKinematic, BindingType::Bool, {})
		DEFINE_ITERATOR(UpdatableComponent)
	}

	struct PhysicsBody::PrivateData
	{
		JPH::BodyID bodyId;
	};

	PhysicsBody::PhysicsBody()
	{
		ZeroMemory(m_PrivateStorage, sizeof(PrivateData));
		m_PrivateData = reinterpret_cast<PrivateData*>(&m_PrivateStorage);
	}

	// ModelImporter will generate prefabs with static PhysicsBodies with mesh shape if it is choosed to

	void PhysicsBody::OnCreate()
	{
		m_Transform = GetTransform();
	}

	void PhysicsBody::OnDestroy()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.DestroyBody(m_PrivateData->bodyId);
		}
	}

	void PhysicsBody::OnEnable()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (!bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.AddBody(m_PrivateData->bodyId, JPH::EActivation::Activate);
		}
	}

	void PhysicsBody::OnDisable()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.RemoveBody(m_PrivateData->bodyId);
		}
	}

	void PhysicsBody::OnUpdate()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (m_PrivateData->bodyId.GetIndex() == 0)
		{
			size_t collidersCount = m_Colliders.size();
			if (collidersCount == 1)
			{
				JPH::Shape* shape = m_Colliders[0]->GetShape();
				if (shape != nullptr)
				{
					Vector3 position = m_Transform->GetPosition();
					Quaternion rotation = m_Transform->GetRotation();
					JPH::EMotionType motionType = m_IsKinematic ? JPH::EMotionType::Kinematic : JPH::EMotionType::Dynamic;
					JPH::BodyCreationSettings settings(shape, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), motionType, 1);
					m_PrivateData->bodyId = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
				}
			}
			else if (collidersCount > 1)
			{
				JPH::StaticCompoundShapeSettings* shapeSettings = new JPH::StaticCompoundShapeSettings();
				for (auto& collider : m_Colliders)
				{
					Transform* colliderTransform = collider->GetTransform();
					if (m_Transform == colliderTransform)
					{
						shapeSettings->AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), collider->GetShape());
					}
					else if (colliderTransform->GetParent() == m_Transform)
					{
						JPH::Shape* shape = collider->GetShape();
						if (shape != nullptr)
						{
							Vector3 position = colliderTransform->GetLocalPosition();
							Quaternion rotation = colliderTransform->GetLocalRotation();
							shapeSettings->AddShape(JPH::Vec3(position.x, position.y, position.z), JPH::QuatArg(rotation.x, rotation.y, rotation.z, rotation.w), shape);
						}
					}
				}
				Vector3 position = m_Transform->GetPosition();
				Quaternion rotation = m_Transform->GetRotation();
				JPH::EMotionType motionType = m_IsKinematic ? JPH::EMotionType::Kinematic : JPH::EMotionType::Dynamic;
				JPH::BodyCreationSettings settings(shapeSettings, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), motionType, 1);
				m_PrivateData->bodyId = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
			}
		}
		else
		{
			JPH::RVec3 position;
			JPH::Quat rotation;
			if (bodyInterface.IsActive(m_PrivateData->bodyId))
			{
				bodyInterface.GetPositionAndRotation(m_PrivateData->bodyId, position, rotation);
				m_Transform->SetPosition(Vector3(position[0], position[1], position[2]));
				m_Transform->SetRotation(Quaternion(rotation.GetX(), rotation.GetY(), rotation.GetZ(), rotation.GetW()));
			}
		}
	}
}
