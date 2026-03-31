#pragma once

namespace JPH
{
	class PhysicsSystem;
	class TempAllocatorImpl;
}

namespace Blueberry
{
	class Physics
	{
	public:
		static bool Initialize();
		static void Shutdown();

		static void Enable();
		static void Disable();

		static void Update(float deltaTime);

	private:
		static JPH::TempAllocatorImpl* s_TempAllocator;
		static JPH::PhysicsSystem* s_PhysicsSystem;

		friend class PhysicsBody;
		friend class Collider;
		friend class CharacterController;
	};
}