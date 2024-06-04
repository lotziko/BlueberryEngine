#include "bbpch.h"
#include "RegisterSceneTypes.h"

#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Core\Object.h"
#include "Components\Component.h"
#include "Components\Transform.h"
#include "Components\Renderer.h"
#include "Components\SpriteRenderer.h"
#include "Components\MeshRenderer.h"
#include "Components\Camera.h"
#include "Components\Light.h"

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
		REGISTER_CLASS(Camera);
		REGISTER_CLASS(Light);
	}
}
