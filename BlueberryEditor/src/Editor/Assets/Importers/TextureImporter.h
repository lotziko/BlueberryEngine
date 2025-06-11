#pragma once
#include "Editor\Assets\AssetImporter.h"
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	enum class TextureImporterType
	{
		Default,
		BaseMap,
		NormalMap,
		Mask,
		Cookie,
	};

	enum class TextureImporterFormat
	{
		None,
		RGBA32,
		RGB24,
		RG16,
		R8,
		BC1,
		BC3,
		BC4,
		BC5,
		BC6H,
		BC7
	};

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

		const TextureImporterType& GetTextureType();
		void SetTextureType(const TextureImporterType& type);

		const TextureImporterFormat& GetTextureFormat();
		void SetTextureFormat(const TextureImporterFormat& format);

		const bool& GetGenerateMipMaps();
		void SetGenerateMipMaps(const bool& generate);

		const bool& IsSRGB();
		void SetSRGB(const bool& srgb);

	protected:
		virtual void ImportData() override;

	private:
		String GetTexturePath();
		TextureFormat GetFormat();

	private:
		bool m_GenerateMipmaps = true;
		bool m_IsSRGB = true;
		WrapMode m_WrapMode = WrapMode::Clamp;
		FilterMode m_FilterMode = FilterMode::Linear;
		TextureImporterShape m_TextureShape = TextureImporterShape::Texture2D;
		TextureImporterType m_TextureType = TextureImporterType::Default;
		TextureImporterFormat m_TextureFormat = TextureImporterFormat::RGBA32;
	};
}