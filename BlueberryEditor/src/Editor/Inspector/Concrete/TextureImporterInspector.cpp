#include "bbpch.h"
#include "TextureImporterInspector.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Assets\ThumbnailCache.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Core\ObjectDB.h"
#include "imgui\imgui.h"

namespace Blueberry
{
	void TextureImporterInspector::Draw(Object* object)
	{
		TextureImporter* textureImporter = static_cast<TextureImporter*>(object);
		textureImporter->ImportDataIfNeeded();

		ObjectInspector::Draw(object);

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
