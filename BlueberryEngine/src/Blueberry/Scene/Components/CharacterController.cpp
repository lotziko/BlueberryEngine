#include "bbpch.h"
#include "CharacterController.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Physics\Physics.h"

#include "Blueberry\Input\Input.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\PhysicsScene.h>
#include <Jolt\Physics\PhysicsSystem.h>
#include <Jolt\Physics\Collision\Shape\CapsuleShape.h>
#include <Jolt\Physics\Collision\Shape\RotatedTranslatedShape.h>
#include <Jolt\Physics\Character\Character.h>

namespace Blueberry
{
	OBJECT_DEFINITION(Component, CharacterController)

	struct CharacterController::CharacterData
	{
		JPH::Ref<JPH::Character> character;
		JPH::RefConst<JPH::Shape> shape;
		JPH::BodyID bodyId;
	};

	CharacterController::~CharacterController()
	{
		if (m_CharacterData != nullptr)
		{
			delete m_CharacterData;
		}
	}

	void CharacterController::OnEnable()
	{
		AddToSceneComponents(UpdatableComponent::Type);
		if (m_CharacterData != nullptr)
		{
			JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
			if (!bodyInterface.IsAdded(m_CharacterData->bodyId))
			{
				bodyInterface.AddBody(m_CharacterData->bodyId, JPH::EActivation::Activate);
			}
		}
	}

	void CharacterController::OnDisable()
	{
		RemoveFromSceneComponents(UpdatableComponent::Type);
		if (m_CharacterData != nullptr)
		{
			JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
			if (bodyInterface.IsAdded(m_CharacterData->bodyId))
			{
				bodyInterface.RemoveBody(m_CharacterData->bodyId);
			}
		}
	}

	void CharacterController::OnUpdate()
	{
		if (m_CharacterData == nullptr)
		{
			m_Transform = GetTransform();
			m_CharacterData = new CharacterData();
			Vector3 position = m_Transform->GetPosition();
			Quaternion rotation = m_Transform->GetRotation();

			m_CharacterData->shape = JPH::RotatedTranslatedShapeSettings(JPH::Vec3(0, 0.5f * m_Height + m_Radius, 0), JPH::Quat::sIdentity(), new JPH::CapsuleShape(0.5f * m_Height, m_Radius)).Create().Get();

			JPH::CharacterSettings settings;
			settings.mLayer = 1;
			settings.mShape = m_CharacterData->shape;
			settings.mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -m_Radius); // Accept contacts that touch the lower sphere of the capsule

			m_CharacterData->character = new JPH::Character(&settings, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), 0, Physics::s_PhysicsSystem);
			m_CharacterData->character->AddToPhysicsSystem();
			m_CharacterData->bodyId = m_CharacterData->character->GetBodyID();
		}
		else
		{
			auto& character = m_CharacterData->character;

			// Input
			{
				float movementSpeed = 5.0f;
				float turnSpeed = 3.0f / 60;

				Vector2 movementAxis = Vector2(Input::IsKeyDown('D') ? 1 : 0 + Input::IsKeyDown('A') ? -1 : 0, Input::IsKeyDown('W') ? 1 : 0 + Input::IsKeyDown('S') ? -1 : 0);
				float turnAxis = Input::IsKeyDown('Q') ? -1 : 0 + Input::IsKeyDown('E') ? 1 : 0;
				/*Vector2 mousePosition = Input::GetMousePosition();
				static Vector2 previousMousePosition;
				float turnAxis = -(mousePosition - previousMousePosition).x;
				previousMousePosition = mousePosition;*/

				Quaternion rotation = m_Transform->GetRotation();
				Vector3 forward = Vector3::Transform(Vector3::UnitZ, rotation);
				Vector3 right = Vector3::Transform(Vector3::UnitX, rotation);
				Vector3 velocity = (right * movementAxis.x + forward * movementAxis.y) * movementSpeed;

				rotation *= Quaternion::CreateFromAxisAngle(Vector3::UnitY, turnAxis * turnSpeed);

				character->SetRotation(JPH::QuatArg(rotation.x, rotation.y, rotation.z, rotation.w));
				character->SetLinearVelocity(character->GetUp() * character->GetLinearVelocity() + JPH::Vec3Arg(velocity.x, velocity.y, velocity.z));
			}

			JPH::RVec3 position;
			JPH::Quat rotation;
			character->GetPositionAndRotation(position, rotation);
			m_Transform->SetPosition(Vector3(position[0], position[1], position[2]));
			m_Transform->SetRotation(Quaternion(rotation.GetX(), rotation.GetY(), rotation.GetZ(), rotation.GetW()));
		}
	}

	const float& CharacterController::GetHeight()
	{
		return m_Height;
	}

	const float& CharacterController::GetRadius()
	{
		return m_Radius;
	}

	void CharacterController::BindProperties()
	{
		BEGIN_OBJECT_BINDING(CharacterController)
		BIND_FIELD(FieldInfo(TO_STRING(m_Entity), &CharacterController::m_Entity, BindingType::ObjectPtr).SetObjectType(Entity::Type))
		BIND_FIELD(FieldInfo(TO_STRING(m_Height), &CharacterController::m_Height, BindingType::Float))
		BIND_FIELD(FieldInfo(TO_STRING(m_Radius), &CharacterController::m_Radius, BindingType::Float))
		END_OBJECT_BINDING()
	}
}