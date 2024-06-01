#pragma once

#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class Texture2D : public Texture
	{
		OBJECT_DECLARATION(Texture2D)
		
	public:
		~Texture2D();

		void Initialize(const TextureProperties& properties);
		void Initialize(const ByteData& byteData);

		static Texture2D* Create(const TextureProperties& properties);

		static void BindProperties();
	};
}