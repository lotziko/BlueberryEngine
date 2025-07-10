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

		static void Update(const float& deltaTime);

	private:
		static inline JPH::TempAllocatorImpl* s_TempAllocator;
		static inline JPH::PhysicsSystem* s_PhysicsSystem = nullptr;

		friend class PhysicsBody;
		friend class CharacterController;
	};
}