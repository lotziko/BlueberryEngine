#pragma once
#include <filesystem>

#include "Editor\Path.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	class AssetImporter;

	class AssetDB
	{
	public:
		struct AssetImportData
		{
			std::filesystem::path relativePath;
			std::string relativePathString;
			long long lastWriteTime;
			bool isDirectory;
		};

	public:
		static void ImportAll();

		static void Import(const std::string& path);

		static AssetImporter* GetImporter(const std::string& path);

		template<class ObjectType>
		static ObjectType* CreateAssetObject(const Guid& guid);

		template<class ObjectType>
		static ObjectType* LoadAssetObject(const Guid& guid);

		static std::string GetAssetCachedDataPath(Object* object, const char* extension);

		static bool HasAssetWithGuidInData(const Guid& guid);
		static void SaveAssetObject(Object* object, const std::string& relativePath);
		static void SaveAssetObjectToCache(Object* object);

	private:
		static void Import(const std::filesystem::path& path);

	public:
		static void Register(const std::string& extension, const std::size_t& importerType);

	private:
		static std::map<std::string, long long> s_PathModifyCache;
		static std::map<std::string, std::size_t> s_ImporterTypes;
		static std::map<std::string, AssetImporter*> s_Importers;
	};

	template<class ObjectType>
	inline ObjectType* AssetDB::CreateAssetObject(const Guid& guid)
	{
		static_assert(std::is_base_of<Object, ObjectType>::value, "Type is not derived from Object.");
		ObjectType* object = Object::Create<ObjectType>();
		ObjectDB::AllocateIdToGuid(object, guid);
		return object;
	}

	template<class ObjectType>
	inline ObjectType* AssetDB::LoadAssetObject(const Guid& guid)
	{
		YamlSerializer serializer;
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		serializer.Deserialize(dataPath.append(guid.ToString().append(".yaml")).string());
		auto& deserializedObjects = serializer.GetDeserializedObjects();
		if (deserializedObjects.size() > 0)
		{
			Object* object = deserializedObjects[0];
			ObjectDB::AllocateIdToGuid(object, guid);
			object->Initialize();
			return (ObjectType*)object;
		}
		return nullptr;
	}

	#define REGISTER_ASSET_IMPORTER( fileExtension, importerType ) AssetDB::Register(fileExtension, importerType);
}