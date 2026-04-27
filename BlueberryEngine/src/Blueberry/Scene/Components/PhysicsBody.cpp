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
		DEFINE_FIELD(PhysicsBody, m_IsKinematic, BindingType::Bool, FieldOptions())
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

	void PhysicsBody::OnFixedUpdate()
	{
		if (m_Transform->GetUpdateCount() > m_UpdateCount)
		{
			JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
			Vector3 position = m_Transform->GetPosition();
			Quaternion rotation = m_Transform->GetRotation();
			bodyInterface.SetPositionAndRotation(m_PrivateData->bodyId, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), JPH::EActivation::Activate);
			m_UpdateCount = m_Transform->GetUpdateCount();
		}
	}

	void PhysicsBody::OnUpdate()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (m_PrivateData->bodyId.GetIndex() == 0)
		{
			// Update is used here because colliders attach to PhysicsBody in OnCreate
			size_t collidersCount = m_Colliders.size();
			if (collidersCount == 1)
			{
				Collider* collider = m_Colliders[0].Get();
				JPH::Shape* shape = collider->GetShape();
				if (shape != nullptr)
				{
					Vector3 position = m_Transform->GetPosition();
					Quaternion rotation = m_Transform->GetRotation();
					JPH::EMotionType motionType = m_IsKinematic ? JPH::EMotionType::Kinematic : JPH::EMotionType::Dynamic;
					JPH::BodyCreationSettings settings(shape, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), motionType, 1);
					settings.mUserData = collider->GetObjectId();
					m_PrivateData->bodyId = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
					collider->m_PhysicsBody = this;
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
						shapeSettings->AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), collider->GetShape(), collider->GetObjectId());
					}
					else if (colliderTransform->GetParent() == m_Transform)
					{
						JPH::Shape* shape = collider->GetShape();
						if (shape != nullptr)
						{
							Vector3 position = colliderTransform->GetLocalPosition();
							Quaternion rotation = colliderTransform->GetLocalRotation();
							shapeSettings->AddShape(JPH::Vec3(position.x, position.y, position.z), JPH::QuatArg(rotation.x, rotation.y, rotation.z, rotation.w), shape, collider->GetObjectId());
							collider->m_PhysicsBody = this;
						}
					}
				}
				Vector3 position = m_Transform->GetPosition();
				Quaternion rotation = m_Transform->GetRotation();
				JPH::EMotionType motionType = m_IsKinematic ? JPH::EMotionType::Kinematic : JPH::EMotionType::Dynamic;
				JPH::BodyCreationSettings settings(shapeSettings, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), motionType, 1);
				m_PrivateData->bodyId = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
			}
			m_UpdateCount = m_Transform->GetUpdateCount();
		}
		else
		{
			if (bodyInterface.IsActive(m_PrivateData->bodyId))
			{
				JPH::RVec3 position;
				JPH::Quat rotation;
				bodyInterface.GetPositionAndRotation(m_PrivateData->bodyId, position, rotation);
				m_Transform->SetPosition(Vector3(position.GetX(), position.GetY(), position.GetZ()));
				m_Transform->SetRotation(Quaternion(rotation.GetX(), rotation.GetY(), rotation.GetZ(), rotation.GetW()));
				m_UpdateCount = m_Transform->GetUpdateCount();
			}
		}
	}

	void PhysicsBody::Move(const Vector3& position)
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.SetPosition(m_PrivateData->bodyId, JPH::RVec3(position.x, position.y, position.z), JPH::EActivation::Activate);
		}
	}

	void PhysicsBody::Move(const Quaternion& rotation)
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.SetRotation(m_PrivateData->bodyId, JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), JPH::EActivation::Activate);
		}
	}

	void PhysicsBody::Move(const Vector3& position, const Quaternion& rotation)
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.SetPositionAndRotation(m_PrivateData->bodyId, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), JPH::EActivation::Activate);
		}
	}

	void PhysicsBody::AddImpulse(const Vector3& impulse)
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.AddImpulse(m_PrivateData->bodyId, JPH::RVec3(impulse.x, impulse.y, impulse.z));
		}
	}

	void PhysicsBody::AddImpulse(const Vector3& impulse, const Vector3& position)
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.AddImpulse(m_PrivateData->bodyId, JPH::RVec3(impulse.x, impulse.y, impulse.z), JPH::RVec3(position.x, position.y, position.z));
		}
	}
}
