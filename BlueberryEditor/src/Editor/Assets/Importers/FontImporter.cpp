#include "FontImporter.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\Font.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\AssetDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(FontImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(FontImporter, AssetImporter)
	}

	void FontImporter::ImportData()
	{
		Guid guid = GetGuid();
		List<Object*> objects;
		String path = GetFilePath();

		List<uint8_t> data;
		FileHelper::Load(data, path);

		size_t fontFileId = TO_HASH("Font");
		Font* font = GetOrCreateAssetObject<Font>(fontFileId);
		SetMainObject(fontFileId);
		font->SetName(GetName());
		font->SetData(data);

		objects.push_back(font);

		AssetDB::SaveAssetObjectsToCache(List<Object*> { font });
	}
}