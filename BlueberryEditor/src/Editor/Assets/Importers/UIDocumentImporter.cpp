#include "UiDocumentImporter.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\UI\UIDocument.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\AssetDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(UIDocumentImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(UIDocumentImporter, AssetImporter)
	}

	void UIDocumentImporter::ImportData()
	{
		Guid guid = GetGuid();
		List<Object*> objects;
		String path = GetFilePath();

		size_t uiDocumentId = TO_HASH("UIDocument");
		UIDocument* document = GetOrCreateAssetObject<UIDocument>(uiDocumentId);
		SetMainObject(uiDocumentId);
		document->SetName(GetName());

		String data = FileHelper::LoadText(path);
		
		size_t i = 0;
		while (i < data.size())
		{
			if (data.compare(i, 6, "image(") == 0)
			{
				size_t start = i + 6;
				size_t end = data.find(')', start);

				if (end == std::string::npos)
				{
					break;
				}

				String inside = data.substr(start, end - start);
				size_t first = inside.find_first_not_of(" \t\n\"'");
				size_t last = inside.find_last_not_of(" \t\n\"'");
				String imagePath = (first != std::string::npos) ? inside.substr(first, last - first + 1) : "";
				AssetImporter* importer = AssetDB::GetImporter(imagePath);
				if (importer != nullptr)
				{
					importer->ImportDataIfNeeded();
					Guid guid = importer->GetGuid();
					data.replace(start + 1, end - start - 2, "guid:" + guid.ToString());
				}
				i = end;
			}
			i += 1;
		}
		document->SetData(data);

		objects.push_back(document);

		AssetDB::SaveAssetObjectsToCache(List<Object*> { document });
	}
}