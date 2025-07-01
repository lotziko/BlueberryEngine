#include "TextureImporterInspector.h"

#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Assets\ThumbnailCache.h"
#include "Editor\Misc\ImGuiHelper.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Core\ObjectDB.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void TextureImporterInspector::Draw(Object* object)
	{
		TextureImporter* textureImporter = static_cast<TextureImporter*>(object);
		textureImporter->ImportDataIfNeeded();

		bool generateMipMaps = textureImporter->GetGenerateMipMaps();
		if (ImGui::BoolEdit("Generate mipmaps", &generateMipMaps))
		{
			textureImporter->SetGenerateMipMaps(generateMipMaps);
		}

		bool srgb = textureImporter->IsSRGB();
		if (ImGui::BoolEdit("Is SRGB", &srgb))
		{
			textureImporter->SetSRGB(srgb);
		}

		WrapMode wrapMode = textureImporter->GetWrapMode();
		int intWrapMode = static_cast<int>(wrapMode);
		static List<String> wrapModes = { "Repeat", "Clamp" };
		if (ImGui::EnumEdit("Wrap Mode", &intWrapMode, &wrapModes))
		{
			wrapMode = static_cast<WrapMode>(intWrapMode);
			textureImporter->SetWrapMode(wrapMode);
		}

		FilterMode filterMode = textureImporter->GetFilterMode();
		int intFilterMode = static_cast<int>(filterMode);
		static List<String> filterModes = { "Point", "Bilinear", "Trilinear", "Anisotropic" };
		if (ImGui::EnumEdit("Filter Mode", &intFilterMode, &filterModes))
		{
			filterMode = static_cast<FilterMode>(intFilterMode);
			textureImporter->SetFilterMode(filterMode);
		}

		TextureImporter::TextureShape shape = textureImporter->GetTextureShape();
		int intShape = static_cast<int>(shape);
		static List<String> shapes = { "Texture2D", "Texture2DArray", "TextureCube", "Texture3D" };
		if (ImGui::EnumEdit("Shape", &intShape, &shapes))
		{
			shape = static_cast<TextureImporter::TextureShape>(intShape);
			textureImporter->SetTextureShape(shape);
		}

		if (shape == TextureImporter::TextureShape::TextureCube)
		{
			TextureImporter::TextureCubeType textureCubeType = textureImporter->GetTextureCubeType();
			int intTextureCubeType = static_cast<int>(textureCubeType);
			static List<String> textureCubeTypes = { "Equirectangular", "Slices" };
			if (ImGui::EnumEdit("TextureCube Type", &intTextureCubeType, &textureCubeTypes))
			{
				textureCubeType = static_cast<TextureImporter::TextureCubeType>(intTextureCubeType);
				textureImporter->SetTextureCubeType(textureCubeType);
			}

			TextureImporter::TextureCubeIBLType textureCubeIBLType = textureImporter->GetTextureCubeIBLType();
			int intTextureCubeIBLType = static_cast<int>(textureCubeIBLType);
			static List<String> textureCubeIBLTypes = { "None", "Specular" };
			if (ImGui::EnumEdit("TextureCube IBL Type", &intTextureCubeIBLType, &textureCubeIBLTypes))
			{
				textureCubeIBLType = static_cast<TextureImporter::TextureCubeIBLType>(intTextureCubeIBLType);
				textureImporter->SetTextureCubeIBLType(textureCubeIBLType);
			}
		}

		TextureImporter::TextureType textureType = textureImporter->GetTextureType();
		int intType = static_cast<int>(textureType);
		static List<String> types = { "Default", "BaseMap", "NormalMap", "Mask", "Cookie" };
		if (ImGui::EnumEdit("Type", &intType, &types))
		{
			textureType = static_cast<TextureImporter::TextureType>(intType);
			textureImporter->SetTextureType(textureType);
		}

		TextureImporter::TextureFormat textureFormat = textureImporter->GetTextureFormat();
		int intFormat = static_cast<int>(textureFormat);
		List<std::pair<String, int>>* formats = nullptr;

		switch (textureType)
		{
		case TextureImporter::TextureType::Default:
		{
			static List<std::pair<String, int>> defaultFormats =
			{
				std::make_pair("RGBA32", static_cast<int>(TextureImporter::TextureFormat::RGBA32)),
				std::make_pair("RGB24", static_cast<int>(TextureImporter::TextureFormat::RGB24)),
				std::make_pair("RG16", static_cast<int>(TextureImporter::TextureFormat::RG16)),
				std::make_pair("R8", static_cast<int>(TextureImporter::TextureFormat::R8)),
				std::make_pair("RGB(A) BC1", static_cast<int>(TextureImporter::TextureFormat::BC1)),
				std::make_pair("RGBA BC3", static_cast<int>(TextureImporter::TextureFormat::BC3)),
				std::make_pair("R BC4", static_cast<int>(TextureImporter::TextureFormat::BC4)),
				std::make_pair("RG BC5", static_cast<int>(TextureImporter::TextureFormat::BC5)),
				std::make_pair("RGB HDR BC6H", static_cast<int>(TextureImporter::TextureFormat::BC6H)),
				std::make_pair("RGB(A) BC7", static_cast<int>(TextureImporter::TextureFormat::BC7))
			};
			formats = &defaultFormats;
		}
		break;
		case TextureImporter::TextureType::BaseMap:
		{
			static List<std::pair<String, int>> baseMapFormats =
			{
				std::make_pair("RGB(Metallness) BC1", static_cast<int>(TextureImporter::TextureFormat::BC1)),
				std::make_pair("RGB(Transparency) BC3", static_cast<int>(TextureImporter::TextureFormat::BC3))
			};
			formats = &baseMapFormats;
		}
		break;
		case TextureImporter::TextureType::NormalMap:
		{
			static List<std::pair<String, int>> normalFormats =
			{
				std::make_pair("Normal(Roughness) BC3", static_cast<int>(TextureImporter::TextureFormat::BC3)),
				std::make_pair("Normal BC5", static_cast<int>(TextureImporter::TextureFormat::BC5)),
				std::make_pair("Normal(Roughness) BC7", static_cast<int>(TextureImporter::TextureFormat::BC7)),
			};
			formats = &normalFormats;
		}
		break;
		case TextureImporter::TextureType::Mask:
		{
			static List<std::pair<String, int>> maskFormats =
			{
				std::make_pair("R8", static_cast<int>(TextureImporter::TextureFormat::R8)),
				std::make_pair("R BC4", static_cast<int>(TextureImporter::TextureFormat::BC4))
			};
			formats = &maskFormats;
		}
		break;
		case TextureImporter::TextureType::Cookie:
		{
			static List<std::pair<String, int>> maskFormats =
			{
				std::make_pair("RGB24", static_cast<int>(TextureImporter::TextureFormat::RGB24)),
				std::make_pair("R8", static_cast<int>(TextureImporter::TextureFormat::R8))
			};
			formats = &maskFormats;
		}
		break;
		}

		if (formats != nullptr)
		{
			if (ImGui::EnumEdit("Format", &intFormat, formats))
			{
				textureImporter->SetTextureFormat(static_cast<TextureImporter::TextureFormat>(intFormat));
			}
		}

		Texture* texture = static_cast<Texture*>(ObjectDB::GetObjectFromGuid(textureImporter->GetGuid(), textureImporter->GetMainObject()));

		if (ImGui::Button("Save"))
		{
			AssetDB::SetDirty(object);
			AssetDB::SaveAssets();
			ThumbnailCache::Refresh(texture);
		}
		
		if (texture->IsClassType(Texture2D::Type))
		{
			ImVec2 size = ImGui::GetContentRegionAvail();
			ImGui::Image(reinterpret_cast<ImTextureID>(texture->GetHandle()), ImVec2(size.x, (texture->GetHeight() * size.x) / static_cast<float>(texture->GetWidth())), ImVec2(0, 1), ImVec2(1, 0));
		}
	}
}
