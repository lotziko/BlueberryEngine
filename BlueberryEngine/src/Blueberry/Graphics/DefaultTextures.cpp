#include "bbpch.h"
#include "DefaultTextures.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	Texture2D* DefaultTextures::GetWhite()
	{
		if (s_WhiteTexture == nullptr)
		{
			const int size = 2;
			byte data[size * size * 4];
			for (int i = 0; i < size * size * 4; i++)
			{
				data[i] = 255;
			}

			TextureProperties properties = {};

			properties.width = size;
			properties.height = size;
			properties.data = data;
			properties.dataSize = size * size * 4;
			properties.format = TextureFormat::R8G8B8A8_UNorm_SRGB;
			properties.wrapMode = WrapMode::Repeat;
			properties.filterMode = FilterMode::Point;
			
			s_WhiteTexture = Texture2D::Create(properties);
		}
		return s_WhiteTexture;
	}
}
