#include "Blueberry\Scene\Components\CharacterController.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Entity.h"

#include "..\..\Physics\Physics.h"
#include "Blueberry\Core\ClassDB.h"
#include "..\..\Input\Input.h"
#include "..\..\Input\Cursor.h"

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
		DEFINE_FIELD(CharacterController, m_CameraTransform, BindingType::ObjectPtr, FieldOptions().SetObjectType(Transform::Type))
		DEFINE_FIELD(CharacterController, m_Height, BindingType::Float, {})
		DEFINE_FIELD(CharacterController, m_Radius, BindingType::Float, {})
	}

	static inline JPH::EBackFaceMode sBackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;
	static inline float		sUpRotationX = 0;
	static inline float		sUpRotationZ = 0;
	static inline float		sMaxSlopeAngle = JPH::DegreesToRadians(45.0f);
	static inline float		sMaxStrength = 100.0f;
	static inline float		sCharacterPadding = 0.02f;
	static inline float		sPenetrationRecoverySpeed = 1.0f;
	static inline float		sPredictiveContactDistance = 0.1f;
	static inline bool		sEnableWalkStairs = true;
	static inline bool		sEnableStickToFloor = true;
	static inline bool		sEnhancedInternalEdgeRemoval = false;
	static inline bool		sCreateInnerBody = false;
	static inline bool		sPlayerCanPushOtherCharacters = true;
	static inline bool		sOtherCharactersCanPushPlayer = true;

	static constexpr float	cCharacterHeightStanding = 1.35f;
	static constexpr float	cCharacterRadiusStanding = 0.3f;
	static constexpr float	cCharacterHeightCrouching = 0.8f;
	static constexpr float	cCharacterRadiusCrouching = 0.3f;
	static constexpr float	cInnerShapeFraction = 0.9f;

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

	void CharacterController::OnEnable()
	{
		AddToSceneComponents(UpdatableComponent::Type);
		if (m_IsInitialized)
		{
			JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
			if (!bodyInterface.IsAdded(m_PrivateData->bodyId))
			{
				bodyInterface.AddBody(m_PrivateData->bodyId, JPH::EActivation::Activate);
			}
		}
	}

	void CharacterController::OnDisable()
	{
		RemoveFromSceneComponents(UpdatableComponent::Type);
		if (m_IsInitialized)
		{
			JPH::BodyInterface& bodyInterface = Physics::s_PhysicsSystem->GetBodyInterface();
			if (bodyInterface.IsAdded(m_PrivateData->bodyId))
			{
				bodyInterface.RemoveBody(m_PrivateData->bodyId);
			}
		}
		Cursor::SetLocked(false);
		Cursor::SetHidden(false);
	}

	void CharacterController::OnUpdate()
	{
		if (!m_IsInitialized)
		{
			m_Transform = GetTransform();
			Vector3 position = m_Transform->GetPosition();
			Quaternion rotation = m_Transform->GetRotation();

			m_PrivateData->shape = JPH::RotatedTranslatedShapeSettings(JPH::Vec3(0, 0.5f * m_Height + m_Radius, 0), JPH::Quat::sIdentity(), new JPH::CapsuleShape(0.5f * m_Height, m_Radius)).Create().Get();

			JPH::CharacterVirtualSettings settings;
			settings.mMaxSlopeAngle = sMaxSlopeAngle;
			settings.mMaxStrength = sMaxStrength;
			settings.mShape = m_PrivateData->shape;
			settings.mMass = 100.0f;
			settings.mBackFaceMode = sBackFaceMode;
			settings.mCharacterPadding = sCharacterPadding;
			settings.mPenetrationRecoverySpeed = sPenetrationRecoverySpeed;
			settings.mPredictiveContactDistance = sPredictiveContactDistance;
			settings.mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -cCharacterRadiusStanding); // Accept contacts that touch the lower sphere of the capsule
			
			m_PrivateData->character = new JPH::CharacterVirtual(&settings, JPH::RVec3(position.x, position.y, position.z), JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w), 0, Physics::s_PhysicsSystem);
			Cursor::SetLocked(true);
			Cursor::SetHidden(true);
			m_IsInitialized = true;
		}
		else
		{
			auto& character = m_PrivateData->character;

			// Input
			{
				float deltaTime = 1.0f / 60.0f;
				float movementSpeed = 10.0f;
				float turnSpeed = 0.15f * deltaTime;
				float surfaceFriction = 5.0f;
				float accelerationMultiplier = 6.0f;

				// Mouse input
				Vector2 turnAxis = Input::GetMouseDelta();

				Quaternion horizontalRotation = m_Transform->GetLocalRotation();
				Quaternion verticalRotation = m_CameraTransform->GetLocalRotation();

				m_Rotation.x += turnAxis.x * turnSpeed;
				m_Rotation.y = std::clamp(m_Rotation.y + turnAxis.y * turnSpeed, ToRadians(-89.0f), ToRadians(89.0f));

				horizontalRotation = Quaternion::CreateFromAxisAngle(Vector3::UnitY, m_Rotation.x);
				verticalRotation = Quaternion::CreateFromAxisAngle(Vector3::UnitX, m_Rotation.y);

				character->SetRotation(JPH::QuatArg(horizontalRotation.x, horizontalRotation.y, horizontalRotation.z, horizontalRotation.w));
				m_CameraTransform->SetLocalRotation(verticalRotation);

				// Movement input
				Vector2 movementAxis = Vector2(static_cast<float>(Input::IsKeyDown('D') ? 1 : 0 + Input::IsKeyDown('A') ? -1 : 0), static_cast<float>(Input::IsKeyDown('W') ? 1 : 0 + Input::IsKeyDown('S') ? -1 : 0));

				Quaternion rotation = m_Transform->GetRotation();
				Vector3 forward = Vector3::Transform(Vector3::UnitZ, rotation);
				Vector3 right = Vector3::Transform(Vector3::UnitX, rotation);
				Vector3 velocity = (right * movementAxis.x + forward * movementAxis.y) * movementSpeed;

				JPH::CharacterVirtual::ExtendedUpdateSettings updateSettings;
				updateSettings.mStickToFloorStepDown = -character->GetUp() * updateSettings.mStickToFloorStepDown.Length();
				updateSettings.mWalkStairsStepUp = character->GetUp() * updateSettings.mWalkStairsStepUp.Length();
				character->ExtendedUpdate(1.0f / 60.0f, -character->GetUp() * Physics::s_PhysicsSystem->GetGravity().Length(), updateSettings, Physics::s_PhysicsSystem->GetDefaultBroadPhaseLayerFilter(Layers::MOVING), Physics::s_PhysicsSystem->GetDefaultLayerFilter(Layers::MOVING), {}, {}, *Physics::s_TempAllocator);
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

				JPH::Vec3 currentVelocity = character->GetUp() * character->GetLinearVelocity() + JPH::Vec3Arg(velocity.x, velocity.y, velocity.z);
				newVelocity += currentVelocity - currentVerticalVelocity;
				character->SetLinearVelocity(newVelocity);
			}

			JPH::RVec3 position;
			JPH::Quat rotation;
			position = character->GetPosition();
			rotation = character->GetRotation();
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
}