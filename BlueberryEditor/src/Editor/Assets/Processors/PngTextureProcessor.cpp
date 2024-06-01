#include "bbpch.h"
#include "PngTextureProcessor.h"

#include "stb\stb_image.h"

namespace Blueberry
{
	PngTextureProcessor::~PngTextureProcessor()
	{
		if (m_Data != nullptr)
		{
			stbi_image_free(m_Data);
		}
	}

	void PngTextureProcessor::Load(const std::string& path)
	{
		stbi_uc* data = nullptr;
		int width, height, channels;

		stbi_set_flip_vertically_on_load(1);
		data = stbi_load(path.c_str(), &width, &height, &channels, 4);
		size_t dataSize = width * height * 4;

		TextureProperties properties = {};

		properties.width = width;
		properties.height = height;
		properties.data = data;
		properties.dataSize = dataSize;

		m_Properties = properties;
	}

	const TextureProperties& PngTextureProcessor::GetProperties()
	{
		return m_Properties;
	}
}
