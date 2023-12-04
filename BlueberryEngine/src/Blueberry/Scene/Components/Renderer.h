#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class Renderer : public Component
	{
		OBJECT_DECLARATION(Renderer)

	public:
		static void BindProperties();
	};
}