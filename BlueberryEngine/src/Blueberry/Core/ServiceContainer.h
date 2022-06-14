#pragma once

#include "Blueberry\Events\Event.h"
#include "Blueberry\Content\ContentManager.h"
#include "Blueberry\Graphics\GraphicsDevice.h"

struct ServiceContainer
{
	static inline Ref<EventDispatcher> EventDispatcher = nullptr;
	static inline Ref<ContentManager> ContentManager = nullptr;
	static inline Ref<GraphicsDevice> GraphicsDevice = nullptr;
	static inline Ref<Renderer2D> Renderer2D = nullptr;
};