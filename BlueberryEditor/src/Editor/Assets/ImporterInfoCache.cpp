#include "ImporterInfoCache.h"

#include "Editor\Path.h"
#include "Editor\Assets\AssetImporter.h"

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"

#include <fstream>

namespace Blueberry
{
	Dictionary<Guid, ImporterInfoCache::ImporterInfo> ImporterInfoCache::s_ImporterInfoCache = {};
	
	void ImporterInfoCache::Load()
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append("ImporterInfoCache");

		if (std::filesystem::exists(dataPath))
		{
			std::ifstream input;
			input.open(dataPath, std::ifstream::binary);
			size_t cacheSize;
			input.read((char*)&cacheSize, sizeof(size_t));

			for (size_t i = 0; i < cacheSize; ++i)
			{
				Guid guid;
				input.read((char*)&guid, sizeof(Guid));
				ImporterInfo info = {};
				info.Read(input);
				s_ImporterInfoCache.insert_or_assign(guid, info);
			}
			input.close();
		}
	}

	void ImporterInfoCache::Save()
	{
		auto dataPath = Path::GetAssetCachePath();
		dataPath.append("ImporterInfoCache");

		size_t cacheSize = s_ImporterInfoCache.size();
		std::ofstream output;
		output.open(dataPath, std::ofstream::binary);
		output.write((char*)&cacheSize, sizeof(size_t));

		for (auto& pair : s_ImporterInfoCache)
		{
			Guid guid = pair.first;
			ImporterInfo info = pair.second;
			output.write((char*)&guid, sizeof(Guid));
			info.Write(output);
		}
		output.close();
	}

	bool ImporterInfoCache::Has(AssetImporter* importer)
	{
		return s_ImporterInfoCache.count(importer->GetGuid()) > 0;
	}

	bool ImporterInfoCache::Get(AssetImporter* importer)
	{
		auto it = s_ImporterInfoCache.find(importer->GetGuid());
		if (it != s_ImporterInfoCache.end())
		{
			ImporterInfo info = it->second;
			Guid guid = importer->GetGuid();
			importer->m_MainObject = info.mainObject;
			for (auto& object : info.objects)
			{
				FileId fileId = std::get<0>(object);
				size_t type = std::get<1>(object);
				String name = std::get<2>(object);

				// Do not create the imported object if it already exists
				if (!ObjectDB::HasGuidAndFileId(guid, fileId))
				{
					ClassDB::ClassInfo classInfo = ClassDB::GetInfo(type);
					Object* importedObject = static_cast<Object*>(classInfo.createInstance());
					importedObject->SetName(name);
					importedObject->SetState(ObjectState::AwaitingLoading);

					ObjectDB::AllocateIdToGuid(importedObject, guid, fileId);
					if (fileId != info.mainObject)
					{
						importer->AddAssetObject(importedObject, fileId);
					}
				}
			}
			return true;
		}
		return false;
	}

	void ImporterInfoCache::Set(AssetImporter* importer)
	{
		Guid guid = importer->GetGuid();
		ImporterInfo info;
		info.mainObject = importer->GetMainObject();
		info.objects.clear();
		
		if (info.mainObject > 0)
		{
			Object* mainObject = ObjectDB::GetObjectFromGuid(guid, info.mainObject);
			info.objects.emplace_back(std::make_tuple(info.mainObject, mainObject->GetType(), mainObject->GetName()));
		}

		for (auto& object : importer->GetAssetObjects())
		{
			Object* assetObject = ObjectDB::GetObject(object.second);
			if (assetObject != nullptr)
			{
				info.objects.emplace_back(std::make_tuple(object.first, assetObject->GetType(), assetObject->GetName()));
			}
		}
		s_ImporterInfoCache.insert_or_assign(guid, info);
	}
}
