#pragma once

#include "Blueberry\Events\Event.h"
#include "Blueberry\Content\ContentManager.h"
#include "Blueberry\Graphics\GraphicsDevice.h"

struct ServiceContainer
{
	Ref<EventDispatcher> EventDispatcher;
	Ref<ContentManager> ContentManager;
	Ref<GraphicsDevice> GraphicsDevice;
	Ref<Renderer2D> Renderer2D;
};