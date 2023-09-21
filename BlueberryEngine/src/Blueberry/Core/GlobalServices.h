#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Events\Event.h"
#include "Blueberry\Content\AssetManager.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Renderer2D.h"

namespace Blueberry
{
	extern EventDispatcher* g_EventDispatcher;
	extern AssetManager* g_AssetManager;
	extern GfxDevice* g_GraphicsDevice;
	extern Renderer2D* g_Renderer2D;
}