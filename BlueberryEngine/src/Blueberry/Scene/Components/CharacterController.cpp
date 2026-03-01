#include "Blueberry\Scene\Components\CharacterController.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Entity.h"

#include "Blueberry\Physics\Physics.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Time.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\PhysicsScene.h>
#include <Jolt\Physics\PhysicsSystem.h>
#include <Jolt\Physics\Collision\Shape\CapsuleShape.h>
#include <Jolt\Physics\Collision\Shape\RotatedTranslatedShape.h>
#include <Jolt\Physics\Character\CharacterVirtual.h>

namespace Layers
{
	static constexpr JPH::ObjectLayer NON_MOVING = 0;
	static constexpr JPH::ObjectLayer MOVING = 1;
	static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

namespace Blueberry
{
	OBJECT_DEFINITION(CharacterController, Component)
	{
		DEFINE_BASE_FIELDS(CharacterController, Component)
		DEFINE_FIELD(CharacterController, m_Height, BindingType::Float, {})
		DEFINE_FIELD(CharacterController, m_Radius, BindingType::Float, {})
		DEFINE_ITERATOR(UpdatableComponent)
	}

	static JPH::EBackFaceMode s_BackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;
	static float s_MaxSlopeAngle = JPH::DegreesToRadians(45.0f);
	static float s_MaxStrength = 100.0f;
	static float s_CharacterPadding = 0.02f;
	static float s_PenetrationRecoverySpeed = 1.0f;
	static float s_PredictiveContactDistance = 0.1f;

	static constexpr float s_CharacterHeightStanding = 1.35f;
	static constexpr float s_CharacterRadiusStanding = 0.3f;
	static constexpr float s_CharacterHeightCrouching = 0.8f;
	static constexpr float s_CharacterRadiusCrouching = 0.3f;
	static constexpr float s_InnerShapeFraction = 0.9f;

	struct CharacterController::PrivateData
	{
		JPH::Ref<JPH::CharacterVirtual> character;
		JPH::RefConst<JPH::Shape> shape;
		JPH::BodyID bodyId;
	};

	CharacterController::CharacterController()
	{
		ZeroMemory(m_PrivateStorage, sizeof(PrivateData));
		m_PrivateData = reinterpret_cast<PrivateData*>(&m_PrivateStorage);
	}

	void CharacterController::OnCreate()
	{
		m_Transform = GetTransform();
		Vector3 position = m_Transform->GetPosition();
		Quaternion rotation = m_Transform->GetRotation();

		m_PrivateData->shape = JPH::RotatedTranslatedShapeSettings(JPH::Vec3(0, 0.5f * m_Height + m_Radius, 0), JPH::Quat::sIdentity(), new JPH::CapsuleShape(0.5f * m_Height, m_Radius)).Create().Get();

		JPH::CharacterVirtualSettings settings;
		settings.mMaxSlopeAngle = s_MaxSlopeAngle;
		settings.mMaxStrength = s_MaxStrength;
		settings.mShape = m_PrivateData->shape;
		settings.mMass = 100.0f;
		settings.mBackFaceMode = s_BackFaceMode;
		settings.mCharacterPadding = s_CharacterPadding;
		settings.mPenetrationRecoverySpeed = s_PenetrationRecoverySpeed;
		settings.mPredictiveContactDistance = s_PredictiveContactDistance;
		settings.mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -s_CharacterRadiusStanding); // Accept contacts that touch the lower sphere of the capsule

		m_PrivateData->character = new JPH::CharacterVirtual(&settings, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), 0, Physics::s_PhysicsSystem);
	}

	void CharacterController::OnDestroy()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.DestroyBody(m_PrivateData->bodyId);
		}
	}

	void CharacterController::OnEnable()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (!bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.AddBody(m_PrivateData->bodyId, JPH::EActivation::Activate);
		}
	}

	void CharacterController::OnDisable()
	{
		JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
		if (bodyInterface.IsAdded(m_PrivateData->bodyId))
		{
			bodyInterface.RemoveBody(m_PrivateData->bodyId);
		}
	}

	void CharacterController::OnFixedUpdate()
	{
		auto& character = m_PrivateData->character;
		float deltaTime = Time::GetFixedDeltaTime();

		JPH::CharacterVirtual::ExtendedUpdateSettings updateSettings;
		updateSettings.mStickToFloorStepDown = -character->GetUp() * updateSettings.mStickToFloorStepDown.Length();
		updateSettings.mWalkStairsStepUp = character->GetUp() * updateSettings.mWalkStairsStepUp.Length();
		character->ExtendedUpdate(deltaTime, -character->GetUp() * Physics::s_PhysicsSystem->GetGravity().Length(), updateSettings, Physics::s_PhysicsSystem->GetDefaultBroadPhaseLayerFilter(Layers::MOVING), Physics::s_PhysicsSystem->GetDefaultLayerFilter(Layers::MOVING), {}, {}, *Physics::s_TempAllocator);
		character->UpdateGroundVelocity();

		JPH::Vec3 currentVerticalVelocity = character->GetLinearVelocity().Dot(character->GetUp()) * character->GetUp();
		JPH::Vec3 groundVelocity = character->GetGroundVelocity();
		JPH::Vec3 newVelocity;
		bool movingTowardsGround = (currentVerticalVelocity.GetY() - groundVelocity.GetY()) < 0.1f;
		if (character->GetGroundState() == JPH::CharacterVirtual::EGroundState::OnGround && !character->IsSlopeTooSteep(character->GetGroundNormal()))
		{
			newVelocity = groundVelocity;
		}
		else
		{
			newVelocity = currentVerticalVelocity;
		}
		newVelocity += (Physics::s_PhysicsSystem->GetGravity()) * deltaTime;

		JPH::Vec3 currentVelocity = character->GetUp() * character->GetLinearVelocity() + JPH::Vec3Arg(m_Velocity.x, m_Velocity.y, m_Velocity.z);
		newVelocity += currentVelocity - currentVerticalVelocity;
		character->SetLinearVelocity(newVelocity);
	}

	void CharacterController::OnUpdate()
	{
		auto& character = m_PrivateData->character;

		JPH::RVec3 position;
		position = character->GetPosition();
		m_Transform->SetPosition(Vector3(position[0], position[1], position[2]));
	}

	const float& CharacterController::GetHeight()
	{
		return m_Height;
	}

	const float& CharacterController::GetRadius()
	{
		return m_Radius;
	}

	void CharacterController::SetVelocity(const Vector3& velocity)
	{
		m_Velocity = velocity;
	}
}