#include "TextureImporterEditor.h"

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
	void TextureImporterEditor::OnEnable()
	{
		m_GenerateMipmapsProperty = m_SerializedObject->FindProperty("m_GenerateMipmaps");
		m_IsSRGBProperty = m_SerializedObject->FindProperty("m_IsSRGB");
		m_WrapModeProperty = m_SerializedObject->FindProperty("m_WrapMode");
		m_FilterModeProperty = m_SerializedObject->FindProperty("m_FilterMode");
		m_TextureShapeProperty = m_SerializedObject->FindProperty("m_TextureShape");
		m_TextureTypeProperty = m_SerializedObject->FindProperty("m_TextureType");
		m_TextureFormatProperty = m_SerializedObject->FindProperty("m_TextureFormat");
		m_TextureCubeTypeProperty = m_SerializedObject->FindProperty("m_TextureCubeType");
		m_TextureCubeIBLTypeProperty = m_SerializedObject->FindProperty("m_TextureCubeIBLType");
	}

	void TextureImporterEditor::OnDrawInspector()
	{
		ImGui::Property(&m_GenerateMipmapsProperty, "Generate mipmaps");
		ImGui::Property(&m_IsSRGBProperty, "Is SRGB");
		ImGui::Property(&m_WrapModeProperty, "Wrap Mode");
		ImGui::Property(&m_FilterModeProperty, "Filter Mode");
		ImGui::Property(&m_TextureShapeProperty, "Shape");

		TextureImporter::TextureShape shape = m_TextureShapeProperty.GetEnum<TextureImporter::TextureShape>();
		if (shape == TextureImporter::TextureShape::TextureCube)
		{
			ImGui::Property(&m_TextureCubeTypeProperty, "TextureCube Type");
			ImGui::Property(&m_TextureCubeIBLTypeProperty, "TextureCube IBL Type");
		}

		ImGui::Property(&m_TextureTypeProperty, "Type");

		if (!m_TextureTypeProperty.IsMixedValue())
		{
			TextureImporter::TextureType textureType = m_TextureTypeProperty.GetEnum<TextureImporter::TextureType>();
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
				int value = m_TextureFormatProperty.GetInt();
				ImGui::SetMixedValue(m_TextureFormatProperty.IsMixedValue());
				if (ImGui::EnumEdit("Format", &value, formats))
				{
					m_TextureFormatProperty.SetInt(value);
				}
				ImGui::SetMixedValue(false);
			}
		}
		m_SerializedObject->ApplyModifiedProperties();

		if (ImGui::Button("Save"))
		{
			for (Object* object : m_SerializedObject->GetTargets())
			{
				AssetDB::SetDirty(object);
			}
			AssetDB::SaveAssets();
		}

		TextureImporter* textureImporter = static_cast<TextureImporter*>(m_SerializedObject->GetTarget());
		Texture* texture = static_cast<Texture*>(ObjectDB::GetObjectFromGuid(textureImporter->GetGuid(), textureImporter->GetMainObject()));
		
		if (texture->IsClassType(Texture2D::Type))
		{
			ImVec2 size = ImGui::GetContentRegionAvail();
			ImGui::Image(reinterpret_cast<ImTextureID>(texture->GetHandle()), ImVec2(size.x, (texture->GetHeight() * size.x) / static_cast<float>(texture->GetWidth())), ImVec2(0, 1), ImVec2(1, 0));
		}
	}
}
