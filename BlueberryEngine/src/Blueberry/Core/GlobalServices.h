#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Events\Event.h"
#include "Blueberry\Content\AssetManager.h"
#include "Blueberry\Graphics\GraphicsDevice.h"

namespace Blueberry
{
	extern EventDispatcher* g_EventDispatcher;
	extern AssetManager* g_AssetManager;
	extern GraphicsDevice* g_GraphicsDevice;
	extern Renderer2D* g_Renderer2D;
}