#include "Blueberry\Core\EngineLayer.h"

#include "..\Scene\RegisterSceneTypes.h"
#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"
#include "Blueberry\Input\Input.h"
#include "Blueberry\Threading\JobSystem.h"
#include "..\Graphics\RegisterGraphicsTypes.h"
#include "Blueberry\Graphics\DefaultShaders.h"
#include "Blueberry\Graphics\Skinning.h"
#include "..\Animations\RegisterAnimationsTypes.h"
#include "Blueberry\Scene\SceneEvents.h"

namespace Blueberry
{
	void EngineLayer::Register()
	{
		RegisterSceneTypes();
		RegisterGraphicsTypes();
		RegisterAnimationsTypes();
	}

	void EngineLayer::Initialize()
	{
		DefaultRenderer::Initialize();
		Input::Initialize();
		JobSystem::Initialize();
		DefaultShaders::Initialize();
		Skinning::Initialize();
	}

	void EngineLayer::Shutdown()
	{
		DefaultRenderer::Shutdown();
		Skinning::Shutdown();
		Input::Shutdown();
	}

	void EngineLayer::Update()
	{
		SceneEvents::Poll();
	}
}