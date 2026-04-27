#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Scene;

	class ProjectBuilder
	{
	public:
		static void Build(Scene* scene, const String& path);
	};
}