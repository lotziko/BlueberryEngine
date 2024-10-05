#include "bbpch.h"
#include "TextureImporterInspector.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Editor\Assets\ThumbnailCache.h"
#include "Blueberry\Graphics\Texture.h"
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
		
		ImVec2 size = ImGui::GetContentRegionAvail();
		ImGui::Image(texture->GetHandle(), ImVec2(size.x, (texture->GetHeight() * size.x) / (float)texture->GetWidth()), ImVec2(0, 1), ImVec2(1, 0));
	}
}
