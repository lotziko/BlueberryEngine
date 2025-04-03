#include "bbpch.h"
#include "DefaultTextures.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	Texture2D* DefaultTextures::GetTexture(const std::string& name)
	{
		if (name == "white")
		{
			return GetWhite();
		}
		if (name == "normal")
		{
			return GetNormal();
		}
		return nullptr;
	}

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

			s_WhiteTexture = Texture2D::Create(size, size, 1, TextureFormat::R8G8B8A8_UNorm_SRGB, WrapMode::Repeat, FilterMode::Point);
			s_WhiteTexture->SetName("White");
			s_WhiteTexture->SetData(data, size * size * 4);
			s_WhiteTexture->Apply();
		}
		return s_WhiteTexture;
	}

	Texture2D* DefaultTextures::GetNormal()
	{
		if (s_NormalTexture == nullptr)
		{
			const int size = 2;
			byte data[size * size * 4];
			for (int i = 0; i < size * size * 4; i += 4)
			{
				data[i] = 127;
				data[i + 1] = 127;
				data[i + 2] = 255;
				data[i + 3] = 255;
			}
			
			s_NormalTexture = Texture2D::Create(size, size, 1, TextureFormat::R8G8B8A8_UNorm, WrapMode::Repeat, FilterMode::Point);
			s_WhiteTexture->SetName("Normal");
			s_NormalTexture->SetData(data, size * size * 4);
			s_NormalTexture->Apply();
		}
		return s_NormalTexture;
	}
}
