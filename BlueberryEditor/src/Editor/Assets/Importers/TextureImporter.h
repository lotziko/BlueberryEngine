#pragma once
#include "Editor\Assets\AssetImporter.h"
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	class TextureImporter : public AssetImporter
	{
		OBJECT_DECLARATION(TextureImporter)

	public:
		TextureImporter() = default;

		static void BindProperties();

	protected:
		virtual void ImportData() override;

	private:
		std::string GetTexturePath();

		bool m_GenerateMipmaps = true;
		bool m_IsSRGB = true;
		WrapMode m_WrapMode = WrapMode::Clamp;
		FilterMode m_FilterMode = FilterMode::Linear;
	};
}