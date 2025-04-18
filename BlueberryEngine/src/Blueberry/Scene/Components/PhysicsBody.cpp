#include "bbpch.h"
#include "PhysicsBody.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Collider.h"
#include "Blueberry\Physics\Physics.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\PhysicsSystem.h>
#include <Jolt\Physics\Collision\Shape\StaticCompoundShape.h>
#include <Jolt\Physics\Body\BodyCreationSettings.h>

namespace Blueberry
{
	OBJECT_DEFINITION(PhysicsBody, Component)
	{
		DEFINE_BASE_FIELDS(PhysicsBody, Component)
		DEFINE_FIELD(PhysicsBody, m_BodyType, BindingType::Enum, FieldOptions().SetEnumHint("Static,Kinematic,Dynamic"))
	}

	// ModelImporter will generate prefabs with static PhysicsBodies with mesh shape if it is choosed to

	void PhysicsBody::OnEnable()
	{
		AddToSceneComponents(UpdatableComponent::Type);
		if (m_IsInitialized)
		{
			JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
			if (!bodyInterface.IsAdded(m_BodyId))
			{
				bodyInterface.AddBody(m_BodyId, JPH::EActivation::Activate);
			}
		}
	}

	void PhysicsBody::OnDisable()
	{
		RemoveFromSceneComponents(UpdatableComponent::Type);
		if (m_IsInitialized)
		{
			JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
			if (bodyInterface.IsAdded(m_BodyId))
			{
				bodyInterface.RemoveBody(m_BodyId);
			}
		}
	}

	void PhysicsBody::OnUpdate()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();

		if (!m_IsInitialized)
		{
			m_Transform = GetTransform();
			Vector3 position = m_Transform->GetLocalPosition();
			Quaternion rotation = m_Transform->GetLocalRotation();
			size_t collidersCount = m_Colliders.size();
			if (collidersCount == 1)
			{
				JPH::Shape* shape;
				shape = m_Colliders[0]->GetShape();
				JPH::EMotionType motionType = static_cast<JPH::EMotionType>(m_BodyType);
				JPH::BodyCreationSettings settings(shape, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), motionType, 1);
				m_BodyId = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
			}
			else if (collidersCount > 1)
			{
				JPH::StaticCompoundShapeSettings* shapeSettings = new JPH::StaticCompoundShapeSettings();
				Transform* bodyTransform = GetTransform();
				for (auto& collider : m_Colliders)
				{
					Transform* colliderTransform = collider->GetTransform();
					if (bodyTransform == colliderTransform)
					{
						shapeSettings->AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), collider->GetShape());
					}
					else if (colliderTransform->GetParent() == bodyTransform)
					{
						Vector3 position = colliderTransform->GetLocalPosition();
						Quaternion rotation = colliderTransform->GetLocalRotation();
						shapeSettings->AddShape(JPH::Vec3(position.x, position.y, position.z), JPH::QuatArg(rotation.x, rotation.y, rotation.z, rotation.w), collider->GetShape());
					}
				}
				JPH::EMotionType motionType = static_cast<JPH::EMotionType>(m_BodyType);
				JPH::BodyCreationSettings settings(shapeSettings, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), motionType, 1);
				m_BodyId = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
			}
			m_IsInitialized = true;
		}
		else if (!m_BodyId.IsInvalid())
		{
			JPH::RVec3 position;
			JPH::Quat rotation;
			if (bodyInterface.IsActive(m_BodyId))
			{
				bodyInterface.GetPositionAndRotation(m_BodyId, position, rotation);
				m_Transform->SetPosition(Vector3(position[0], position[1], position[2]));
				m_Transform->SetRotation(Quaternion(rotation.GetX(), rotation.GetY(), rotation.GetZ(), rotation.GetW()));
			}
		}
	}
}
