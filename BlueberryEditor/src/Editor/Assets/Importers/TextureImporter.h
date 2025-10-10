#pragma once
#include "Editor\Assets\AssetImporter.h"
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	class TextureImporter : public AssetImporter
	{
		OBJECT_DECLARATION(TextureImporter)

	public:
		enum class TextureType
		{
			Default,
			BaseMap,
			NormalMap,
			Mask,
			Cookie,
		};

		enum class TextureFormat
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

		enum class TextureShape
		{
			Texture2D,
			Texture2DArray,
			TextureCube,
			Texture3D
		};

		enum class TextureCubeType
		{
			Equirectangular,
			Slices
		};

		enum class TextureCubeIBLType
		{
			None,
			Specular
		};

	public:
		TextureImporter() = default;

		const TextureShape& GetTextureShape();
		void SetTextureShape(const TextureShape& shape);

		const TextureType& GetTextureType();
		void SetTextureType(const TextureType& type);

		const TextureFormat& GetTextureFormat();
		void SetTextureFormat(const TextureFormat& format);

		const TextureCubeType& GetTextureCubeType();
		void SetTextureCubeType(const TextureCubeType& cubeType);

		const TextureCubeIBLType& GetTextureCubeIBLType();
		void SetTextureCubeIBLType(const TextureCubeIBLType& cubeIBLType);

		const bool& GetGenerateMipMaps();
		void SetGenerateMipMaps(const bool& generate);

		const bool& IsSRGB();
		void SetSRGB(const bool& srgb);

		const WrapMode& GetWrapMode();
		void SetWrapMode(const WrapMode& wrapMode);

		const FilterMode& GetFilterMode();
		void SetFilterMode(const FilterMode& filterMode);

		static String GetTexturePath(const Guid& guid);

	protected:
		virtual void ImportData() override;

	private:
		Blueberry::TextureFormat GetFormat();

	private:
		bool m_GenerateMipmaps = true;
		bool m_IsSRGB = true;
		WrapMode m_WrapMode = WrapMode::Clamp;
		FilterMode m_FilterMode = FilterMode::Bilinear;
		TextureShape m_TextureShape = TextureShape::Texture2D;
		TextureType m_TextureType = TextureType::Default;
		TextureFormat m_TextureFormat = TextureFormat::RGBA32;
		TextureCubeType m_TextureCubeType = TextureCubeType::Equirectangular;
		TextureCubeIBLType m_TextureCubeIBLType = TextureCubeIBLType::None;
	};
}