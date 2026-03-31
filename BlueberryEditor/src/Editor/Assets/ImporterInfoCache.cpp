#include "ImporterInfoCache.h"

#include "Editor\Path.h"
#include "Editor\Assets\AssetImporter.h"
#include "Editor\Misc\PathHelper.h"

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"

#include <fstream>

namespace Blueberry
{
	Dictionary<Guid, ImporterInfo> ImporterInfoCache::s_ImporterInfoCache = {};
	
	void ImporterInfoCache::Load()
	{
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		dataPath.append("ImporterInfoCache");

		if (std::filesystem::exists(dataPath))
		{
			std::ifstream input;
			input.open(dataPath, std::ios::in | std::ifstream::binary);
			uint32_t version;
			size_t cacheSize;
			input.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
			input.read(reinterpret_cast<char*>(&cacheSize), sizeof(size_t));

			for (size_t i = 0; i < cacheSize; ++i)
			{
				if (input.eof())
				{
					break;
				}

				Guid guid;
				input.read(reinterpret_cast<char*>(&guid), sizeof(Guid));
				ImporterInfo info = {};
				Read(input, info);
				s_ImporterInfoCache.insert_or_assign(guid, info);
			}
			input.close();
		}
	}

	void ImporterInfoCache::Save()
	{
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		dataPath.append("ImporterInfoCache");

		std::ofstream output;
		output.open(dataPath, std::ios::out | std::ofstream::binary);
		if (output.is_open())
		{
			uint32_t version = 1;
			size_t cacheSize = s_ImporterInfoCache.size();
			output.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));
			output.write(reinterpret_cast<char*>(&cacheSize), sizeof(size_t));

			for (auto& pair : s_ImporterInfoCache)
			{
				Guid guid = pair.first;
				ImporterInfo info = pair.second;
				output.write(reinterpret_cast<char*>(&guid), sizeof(Guid));
				Write(output, info);
			}
			output.close();
		}
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
				TypeId typeId = std::get<1>(object);
				String name = std::get<2>(object);

				// Do not create the imported object if it already exists
				if (!ObjectDB::HasGuidAndFileId(guid, fileId))
				{
					const ClassInfo* classInfo = ClassDB::GetInfo(typeId);
					if (classInfo == nullptr)
					{
						BB_ERROR("Class not exists.");
						return false;
					}
					Object* importedObject = static_cast<Object*>(classInfo->Create());
					importedObject->SetName(name);
					importedObject->SetState(ObjectState::AwaitingLoading);

					ObjectDB::AllocateIdToGuid(importedObject, guid, fileId);
					if (fileId != info.mainObject)
					{
						importer->AddAssetObject(importedObject, fileId);
					}
				}
			}
			importer->m_LastWrite = info.lastWrite;
			return true;
		}
		return false;
	}

	void ImporterInfoCache::Set(AssetImporter* importer)
	{
		Guid guid = importer->GetGuid();
		ImporterInfo info = {};
		info.mainObject = importer->GetMainObject();
		info.objects.clear();
		
		if (info.mainObject > 0)
		{
			Object* mainObject = ObjectDB::GetObjectFromGuid(guid, info.mainObject);
			if (mainObject != nullptr)
			{
				info.objects.push_back(std::make_tuple(info.mainObject, mainObject->GetType(), mainObject->GetName()));
			}
		}

		for (auto& object : importer->GetAssetObjects())
		{
			Object* assetObject = ObjectDB::GetObject(object.second);
			if (assetObject != nullptr)
			{
				info.objects.push_back(std::make_tuple(object.first, assetObject->GetType(), assetObject->GetName()));
			}
		}

		info.lastWrite = std::max(PathHelper::GetLastWriteTime(importer->GetFilePath()), PathHelper::GetLastWriteTime(importer->GetMetaFilePath()));
		s_ImporterInfoCache.insert_or_assign(guid, info);
	}

	void ImporterInfoCache::Clear(AssetImporter* importer)
	{
		s_ImporterInfoCache.erase(importer->GetGuid());
	}

	void ImporterInfoCache::Read(std::istream& is, ImporterInfo& info)
	{
		is.read(reinterpret_cast<char*>(&info.mainObject), sizeof(FileId));
		size_t objectCount;
		is.read(reinterpret_cast<char*>(&objectCount), sizeof(size_t));

		for (size_t i = 0; i < objectCount; ++i)
		{
			if (is.eof())
			{
				break;
			}

			FileId objectFileId;
			is.read(reinterpret_cast<char*>(&objectFileId), sizeof(FileId));

			size_t typeNameSize;
			is.read(reinterpret_cast<char*>(&typeNameSize), sizeof(size_t));
			String typeName(typeNameSize, ' ');
			is.read(typeName.data(), typeNameSize);

			size_t nameSize;
			is.read(reinterpret_cast<char*>(&nameSize), sizeof(size_t));
			String name(nameSize, ' ');
			is.read(name.data(), nameSize);

			info.objects.push_back(std::make_tuple(objectFileId, ClassDB::GetTypeId(typeName), name));
		}

		is.read(reinterpret_cast<char*>(&info.lastWrite), sizeof(long long));
	}

	void ImporterInfoCache::Write(std::ostream& os, ImporterInfo& info)
	{
		os.write(reinterpret_cast<char*>(&info.mainObject), sizeof(FileId));

		size_t objectCount = info.objects.size();
		os.write(reinterpret_cast<char*>(&objectCount), sizeof(size_t));

		for (auto& tuple : info.objects)
		{
			FileId objectFileId = std::get<0>(tuple);
			os.write(reinterpret_cast<char*>(&objectFileId), sizeof(FileId));

			String typeName = ClassDB::GetInfo(std::get<1>(tuple))->name;
			size_t typeNameSize = typeName.size();
			os.write(reinterpret_cast<char*>(&typeNameSize), sizeof(size_t));
			os.write(typeName.data(), typeNameSize);

			String name = std::get<2>(tuple);
			size_t nameSize = name.size();
			os.write(reinterpret_cast<char*>(&nameSize), sizeof(size_t));
			os.write(name.data(), nameSize);
		}

		os.write(reinterpret_cast<char*>(&info.lastWrite), sizeof(long long));
	}
}
