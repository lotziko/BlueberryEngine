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

		TextureImporterType textureType = textureImporter->GetTextureType();
		int intType = static_cast<int>(textureType);
		static List<String> types = { "Default", "BaseMap", "NormalMap", "Mask", "Cookie" };
		if (ImGui::EnumEdit("Type", &intType, &types))
		{
			textureType = static_cast<TextureImporterType>(intType);
			textureImporter->SetTextureType(textureType);
		}

		TextureImporterFormat textureFormat = textureImporter->GetTextureFormat();
		int intFormat = static_cast<int>(textureFormat);
		List<std::pair<String, int>>* formats = nullptr;

		switch (textureType)
		{
		case TextureImporterType::Default:
		{
			static List<std::pair<String, int>> defaultFormats =
			{
				std::make_pair("RGBA32", static_cast<int>(TextureImporterFormat::RGBA32)),
				std::make_pair("RGB24", static_cast<int>(TextureImporterFormat::RGB24)),
				std::make_pair("RG16", static_cast<int>(TextureImporterFormat::RG16)),
				std::make_pair("R8", static_cast<int>(TextureImporterFormat::R8)),
				std::make_pair("RGB(A) BC1", static_cast<int>(TextureImporterFormat::BC1)),
				std::make_pair("RGBA BC3", static_cast<int>(TextureImporterFormat::BC3)),
				std::make_pair("R BC4", static_cast<int>(TextureImporterFormat::BC4)),
				std::make_pair("RG BC5", static_cast<int>(TextureImporterFormat::BC5)),
				std::make_pair("RGB HDR BC6H", static_cast<int>(TextureImporterFormat::BC6H)),
				std::make_pair("RGB(A) BC7", static_cast<int>(TextureImporterFormat::BC7))
			};
			formats = &defaultFormats;
		}
		break;
		case TextureImporterType::BaseMap:
		{
			static List<std::pair<String, int>> baseMapFormats =
			{
				std::make_pair("RGB(Metallness) BC1", static_cast<int>(TextureImporterFormat::BC1)),
				std::make_pair("RGB(Transparency) BC3", static_cast<int>(TextureImporterFormat::BC3))
			};
			formats = &baseMapFormats;
		}
		break;
		case TextureImporterType::NormalMap:
		{
			static List<std::pair<String, int>> normalFormats =
			{
				std::make_pair("Normal(Roughness) BC3", static_cast<int>(TextureImporterFormat::BC3)),
				std::make_pair("Normal BC5", static_cast<int>(TextureImporterFormat::BC5)),
				std::make_pair("Normal(Roughness) BC7", static_cast<int>(TextureImporterFormat::BC7)),
			};
			formats = &normalFormats;
		}
		break;
		case TextureImporterType::Mask:
		{
			static List<std::pair<String, int>> maskFormats =
			{
				std::make_pair("R8", static_cast<int>(TextureImporterFormat::R8)),
				std::make_pair("R BC4", static_cast<int>(TextureImporterFormat::BC4))
			};
			formats = &maskFormats;
		}
		break;
		case TextureImporterType::Cookie:
		{
			static List<std::pair<String, int>> maskFormats =
			{
				std::make_pair("RGB24", static_cast<int>(TextureImporterFormat::RGB24)),
				std::make_pair("R8", static_cast<int>(TextureImporterFormat::R8))
			};
			formats = &maskFormats;
		}
		break;
		}

		if (formats != nullptr)
		{
			if (ImGui::EnumEdit("Format", &intFormat, formats))
			{
				textureImporter->SetTextureFormat(static_cast<TextureImporterFormat>(intFormat));
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
