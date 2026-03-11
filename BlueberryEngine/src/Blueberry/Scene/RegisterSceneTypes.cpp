#include "RegisterSceneTypes.h"

#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Core\Object.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Renderer.h"
#include "Blueberry\Scene\Components\SpriteRenderer.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\Animator.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\Light.h"
#include "Blueberry\Scene\Components\ProbeVolume.h"
#include "Blueberry\Scene\Components\ReflectionProbe.h"
#include "Blueberry\Scene\Components\PhysicsBody.h"
#include "Blueberry\Scene\Components\Collider.h"
#include "Blueberry\Scene\Components\BoxCollider.h"
#include "Blueberry\Scene\Components\SphereCollider.h"
#include "Blueberry\Scene\Components\MeshCollider.h"
#include "Blueberry\Scene\Components\CharacterController.h"

namespace Blueberry
{
	void RegisterSceneTypes()
	{
		REGISTER_ABSTRACT_CLASS(Object);
		REGISTER_CLASS(Entity);
		REGISTER_ABSTRACT_CLASS(Component);
		REGISTER_ITERATOR(UpdatableComponent);
		REGISTER_CLASS(Transform);
		REGISTER_ABSTRACT_CLASS(Renderer);
		REGISTER_CLASS(SpriteRenderer);
		REGISTER_CLASS(MeshRenderer);
		REGISTER_CLASS(SkinnedMeshRenderer);
		REGISTER_CLASS(SkyRenderer);
		REGISTER_CLASS(Animator);
		REGISTER_CLASS(Camera);
		REGISTER_CLASS(Light);
		REGISTER_CLASS(ProbeVolume);
		REGISTER_CLASS(ReflectionProbe);
		REGISTER_CLASS(PhysicsBody);
		REGISTER_ABSTRACT_CLASS(Collider);
		REGISTER_CLASS(BoxCollider);
		REGISTER_CLASS(SphereCollider);
		REGISTER_CLASS(MeshCollider);
		REGISTER_CLASS(CharacterController);
	}
}
