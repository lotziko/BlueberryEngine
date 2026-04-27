#pragma once

#include "Blueberry\Core\Base.h"

namespace JPH
{
	class PhysicsSystem;
	class TempAllocatorImpl;
}

namespace Blueberry
{
	class Collider;

	struct RaycastHitData
	{
		Vector3 position;
		Collider* collider;
	};

	class BB_API Physics
	{
	public:
		static bool Initialize();
		static void Shutdown();

		static void Enable();
		static void Disable();

		static void Update(float deltaTime);

		static bool Raycast(const Vector3& origin, const Vector3& direction, float distance, RaycastHitData& raycastHitData);

	private:
		static JPH::TempAllocatorImpl* s_TempAllocator;
		static JPH::PhysicsSystem* s_PhysicsSystem;

		friend class PhysicsBody;
		friend class Collider;
		friend class CharacterController;
#ifdef JPH_DEBUG_RENDERER
		friend class PhysicsGizmoRenderer;
#endif
	};
}