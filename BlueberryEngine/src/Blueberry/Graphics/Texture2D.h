#pragma once
#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class Texture2D : public Texture
	{
		OBJECT_DECLARATION(Texture2D)

	public:
		Texture2D() = default;
		Texture2D(const TextureProperties& properties);

		static Ref<Texture2D> Create(const TextureProperties& properties);
	};
}