#pragma once
#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class Texture2D : public Texture
	{
		OBJECT_DECLARATION(Texture2D)

	public:
		Texture2D() = default;
		Texture2D(const std::string& path);

		static Ref<Texture2D> Create(const std::string& path);
	};
}