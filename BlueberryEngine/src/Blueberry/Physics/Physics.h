#pragma once

namespace JPH
{
	class PhysicsSystem;
}

namespace Blueberry
{
	class Physics
	{
	public:
		static bool Initialize();
		static void Shutdown();

		static void Update(const float& deltaTime);

	private:
		static inline JPH::PhysicsSystem* s_PhysicsSystem = nullptr;

		friend class PhysicsBody;
		friend class CharacterController;
	};
}