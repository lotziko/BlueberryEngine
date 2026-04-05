#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Scene;

	class RuntimeLoader
	{
	public:
		static void Load(Scene* scene);
	};
}