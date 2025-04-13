#include "bbpch.h"
#include "RegisterSceneTypes.h"

#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Core\Object.h"
#include "Components\Component.h"
#include "Components\Transform.h"
#include "Components\Renderer.h"
#include "Components\SpriteRenderer.h"
#include "Components\MeshRenderer.h"
#include "Components\SkyRenderer.h"
#include "Components\Camera.h"
#include "Components\Light.h"
#include "Components\PhysicsBody.h"
#include "Components\Collider.h"
#include "Components\BoxCollider.h"
#include "Components\SphereCollider.h"
#include "Components\CharacterController.h"

namespace Blueberry
{
	void RegisterSceneTypes()
	{
		REGISTER_ABSTRACT_CLASS(Object);
		REGISTER_ABSTRACT_CLASS(Component);
		REGISTER_CLASS(Entity);
		REGISTER_CLASS(Transform);
		REGISTER_ABSTRACT_CLASS(Renderer);
		REGISTER_CLASS(SpriteRenderer);
		REGISTER_CLASS(MeshRenderer);
		REGISTER_CLASS(SkyRenderer);
		REGISTER_CLASS(Camera);
		REGISTER_CLASS(Light);
		REGISTER_CLASS(PhysicsBody);
		REGISTER_ABSTRACT_CLASS(Collider);
		REGISTER_CLASS(BoxCollider);
		REGISTER_CLASS(SphereCollider);
		REGISTER_CLASS(CharacterController);
	}
}
