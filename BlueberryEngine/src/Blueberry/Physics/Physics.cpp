#include "Physics.h"

#include <Jolt\Jolt.h>

#include <Jolt\RegisterTypes.h>
#include <Jolt\Core\Factory.h>
#include <Jolt\Core\TempAllocator.h>
#include <Jolt\Core\JobSystemThreadPool.h>
#include <Jolt\Physics\PhysicsSettings.h>
#include <Jolt\Physics\PhysicsSystem.h>

#include <Jolt\Physics\Collision\Shape\BoxShape.h>
#include <Jolt\Physics\Body\BodyCreationSettings.h>

using namespace JPH;

namespace Layers
{
	static constexpr ObjectLayer NON_MOVING = 0;
	static constexpr ObjectLayer MOVING = 1;
	static constexpr ObjectLayer NUM_LAYERS = 2;
};

class ObjectLayerPairFilterImpl : public ObjectLayerPairFilter
{
public:
	virtual bool ShouldCollide(ObjectLayer inObject1, ObjectLayer inObject2) const override
	{
		switch (inObject1)
		{
		case Layers::NON_MOVING:
			return inObject2 == Layers::MOVING; // Non moving only collides with moving
		case Layers::MOVING:
			return true; // Moving collides with everything
		default:
			JPH_ASSERT(false);
			return false;
		}
	}
};

namespace BroadPhaseLayers
{
	static constexpr BroadPhaseLayer NON_MOVING(0);
	static constexpr BroadPhaseLayer MOVING(1);
	static constexpr uint NUM_LAYERS(2);
};

class BPLayerInterfaceImpl final : public BroadPhaseLayerInterface
{
public:
	BPLayerInterfaceImpl()
	{
		// Create a mapping table from object to broad phase layer
		mObjectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
		mObjectToBroadPhase[Layers::MOVING] = BroadPhaseLayers::MOVING;
	}

	virtual uint GetNumBroadPhaseLayers() const override
	{
		return BroadPhaseLayers::NUM_LAYERS;
	}

	virtual BroadPhaseLayer	GetBroadPhaseLayer(ObjectLayer inLayer) const override
	{
		JPH_ASSERT(inLayer < Layers::NUM_LAYERS);
		return mObjectToBroadPhase[inLayer];
	}

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
	virtual const char *			GetBroadPhaseLayerName(BroadPhaseLayer inLayer) const override
	{
		switch ((BroadPhaseLayer::Type)inLayer)
		{
		case (BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING:	return "NON_MOVING";
		case (BroadPhaseLayer::Type)BroadPhaseLayers::MOVING:		return "MOVING";
		default:													JPH_ASSERT(false); return "INVALID";
		}
	}
#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED

private:
	BroadPhaseLayer	mObjectToBroadPhase[Layers::NUM_LAYERS];
};

class ObjectVsBroadPhaseLayerFilterImpl : public ObjectVsBroadPhaseLayerFilter
{
public:
	virtual bool ShouldCollide(ObjectLayer inLayer1, BroadPhaseLayer inLayer2) const override
	{
		switch (inLayer1)
		{
		case Layers::NON_MOVING:
			return inLayer2 == BroadPhaseLayers::MOVING;
		case Layers::MOVING:
			return true;
		default:
			JPH_ASSERT(false);
			return false;
		}
	}
};

namespace Blueberry
{
	JobSystemThreadPool* job_system;

	BPLayerInterfaceImpl broad_phase_layer_interface;
	ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
	ObjectLayerPairFilterImpl object_vs_object_layer_filter;

	bool Physics::Initialize()
	{
		RegisterDefaultAllocator();

		Factory::sInstance = new JPH::Factory();
		RegisterTypes();
		return true;
	}

	void Physics::Shutdown()
	{
		UnregisterTypes();
		delete Factory::sInstance;
		Factory::sInstance = nullptr;
	}

	void Physics::Enable()
	{
		s_TempAllocator = new TempAllocatorImpl(10 * 1024 * 1024);
		job_system = new JobSystemThreadPool(cMaxPhysicsJobs, cMaxPhysicsBarriers, thread::hardware_concurrency() - 1);

		const uint cMaxBodies = 1024;
		const uint cNumBodyMutexes = 0;
		const uint cMaxBodyPairs = 1024;
		const uint cMaxContactConstraints = 1024;

		s_PhysicsSystem = new PhysicsSystem();
		s_PhysicsSystem->Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints, broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);

		// Increase sleep threshold
		PhysicsSettings settings = s_PhysicsSystem->GetPhysicsSettings();
		settings.mPointVelocitySleepThreshold /= 4;
		s_PhysicsSystem->SetPhysicsSettings(settings);

		BoxShapeSettings floorShapeSettings(Vec3(100.0f, 1.0f, 100.0f));
		ShapeSettings::ShapeResult floor_shape_result = floorShapeSettings.Create();
		ShapeRefC floor_shape = floor_shape_result.Get();
		BodyCreationSettings floor_settings(floor_shape, RVec3(0.0f, -1.0f, 0.0f), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
		Body *floor = s_PhysicsSystem->GetBodyInterface().CreateBody(floor_settings);
		s_PhysicsSystem->GetBodyInterface().AddBody(floor->GetID(), EActivation::DontActivate);
	}

	void Physics::Disable()
	{
	}

	void Physics::Update(const float& deltaTime)
	{
		const int cCollisionSteps = 1;
		s_PhysicsSystem->Update(deltaTime, cCollisionSteps, s_TempAllocator, job_system);
	}
}
