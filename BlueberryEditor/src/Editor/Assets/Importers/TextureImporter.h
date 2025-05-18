#pragma once
#include "Editor\Assets\AssetImporter.h"
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	enum class TextureImporterShape
	{
		Texture2D,
		Texture2DArray,
		TextureCube,
		Texture3D
	};

	class TextureImporter : public AssetImporter
	{
		OBJECT_DECLARATION(TextureImporter)

	public:
		TextureImporter() = default;

	protected:
		virtual void ImportData() override;

	private:
		String GetTexturePath();

		bool m_GenerateMipmaps = true;
		bool m_IsSRGB = true;
		WrapMode m_WrapMode = WrapMode::Clamp;
		FilterMode m_FilterMode = FilterMode::Linear;
		TextureImporterShape m_TextureShape = TextureImporterShape::Texture2D;
	};
}