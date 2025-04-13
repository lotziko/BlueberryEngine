#include "bbpch.h"
#include "DefaultTextures.h"

#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	Texture* DefaultTextures::GetTexture(const std::string& name, const TextureDimension& dimension)
	{
		if (dimension == TextureDimension::Texture2D)
		{
			if (name == "white")
			{
				return GetWhite2D();
			}
			if (name == "normal")
			{
				return GetNormal2D();
			}
		}
		else if (dimension == TextureDimension::TextureCube)
		{
			if (name == "white")
			{
				return GetWhiteCube();
			}
		}
		return nullptr;
	}

	Texture2D* DefaultTextures::GetWhite2D()
	{
		if (s_WhiteTexture2D == nullptr)
		{
			const uint32_t size = 2;
			uint8_t data[size * size * 4];
			for (uint32_t i = 0; i < size * size * 4; i++)
			{
				data[i] = 255;
			}

			s_WhiteTexture2D = Texture2D::Create(size, size, 1, TextureFormat::R8G8B8A8_UNorm_SRGB, WrapMode::Repeat, FilterMode::Point);
			s_WhiteTexture2D->SetName("White");
			s_WhiteTexture2D->SetData(data, size * size * 4);
			s_WhiteTexture2D->Apply();
		}
		return s_WhiteTexture2D;
	}

	Texture2D* DefaultTextures::GetNormal2D()
	{
		if (s_NormalTexture2D == nullptr)
		{
			const uint32_t size = 2;
			uint8_t data[size * size * 4];
			for (uint32_t i = 0; i < size * size * 4; i += 4)
			{
				data[i] = 127;
				data[i + 1] = 127;
				data[i + 2] = 255;
				data[i + 3] = 255;
			}
			
			s_NormalTexture2D = Texture2D::Create(size, size, 1, TextureFormat::R8G8B8A8_UNorm, WrapMode::Repeat, FilterMode::Point);
			s_NormalTexture2D->SetName("Normal");
			s_NormalTexture2D->SetData(data, size * size * 4);
			s_NormalTexture2D->Apply();
		}
		return s_NormalTexture2D;
	}

	TextureCube* DefaultTextures::GetWhiteCube()
	{
		if (s_WhiteTextureCube == nullptr)
		{
			const uint32_t size = 6 * 4;
			uint8_t data[size];
			for (int i = 0; i < size; ++i)
			{
				data[i] = 255;
			}
			s_WhiteTextureCube = TextureCube::Create(1, 1, 1, TextureFormat::R8G8B8A8_UNorm, WrapMode::Repeat, FilterMode::Point);
			s_WhiteTextureCube->SetName("White");
			s_WhiteTextureCube->SetData(data, size);
			s_WhiteTextureCube->Apply();
		}
		return s_WhiteTextureCube;
	}
}
