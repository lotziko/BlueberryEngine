#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class TextureCube;

	class ReflectionGenerator
	{
	public:
		static TextureCube* GenerateReflectionTexture(TextureCube* source);
	};
}