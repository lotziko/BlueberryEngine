#pragma once

#include "Blueberry\Graphics\Structs.h"

namespace Blueberry
{
	class PngTextureProcessor
	{
		typedef unsigned char stbi_uc;

	public:
		PngTextureProcessor() = default;
		~PngTextureProcessor();

		void Load(const std::string& path);
		const TextureProperties& GetProperties();

	private:
		stbi_uc* m_Data = nullptr;
		TextureProperties m_Properties;
	};
}