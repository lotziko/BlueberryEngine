#include "bbpch.h"
#include "TextureImporterInspector.h"
#include "Editor\Assets\Importers\TextureImporter.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Core\ObjectDB.h"
#include "imgui\imgui.h"

namespace Blueberry
{
	void TextureImporterInspector::Draw(Object* object)
	{
		TextureImporter* textureImporter = static_cast<TextureImporter*>(object);
		textureImporter->ImportDataIfNeeded();
		auto& objects = textureImporter->GetImportedObjects();
		Texture* texture = static_cast<Texture*>(ObjectDB::GetObject(objects.begin()->second));
		ImVec2 size = ImGui::GetContentRegionAvail();
		ImGui::Image(texture->GetHandle(), ImVec2(size.x, (texture->GetHeight() * size.x) / (float)texture->GetWidth()), ImVec2(0, 1), ImVec2(1, 0));
	}
}
