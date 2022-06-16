#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Events\Event.h"
#include "Blueberry\Content\ContentManager.h"
#include "Blueberry\Graphics\GraphicsDevice.h"

namespace Blueberry
{
	extern EventDispatcher* g_EventDispatcher;
	extern ContentManager* g_ContentManager;
	extern GraphicsDevice* g_GraphicsDevice;
	extern Renderer2D* g_Renderer2D;
}