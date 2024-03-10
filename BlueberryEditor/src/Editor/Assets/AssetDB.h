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
		static void Refresh();

		static AssetImporter* Import(const std::string& relativePath);
		static AssetImporter* Import(const Guid& guid);

		static AssetImporter* GetImporter(const std::string& path);

		template<class ObjectType>
		static ObjectType* CreateAssetObject(const Guid& guid);

		template<class ObjectType>
		static ObjectType* LoadAssetObject(const Guid& guid);

		static std::string GetAssetCachedDataPath(Object* object, const char* extension);

		static bool HasAssetWithGuidInData(const Guid& guid);
		static void CreateAsset(Object* object, const std::string& relativePath);
		static void SaveAssetObjectToCache(Object* object);
		static void SetDirty(Object* object);
		static void DeleteAssetFromData(const Guid& guid);
		static void SaveAssets();

	private:
		static AssetImporter* CreateImporter(const std::filesystem::path& path);

	public:
		static void Register(const std::string& extension, const std::size_t& importerType);

	private:
		static std::map<std::string, std::size_t> s_ImporterTypes;
		static std::map<std::string, AssetImporter*> s_Importers;
		static std::map<Guid, std::string> s_GuidToPath;
		static std::vector<ObjectId> s_DirtyAssets;

		// Save these to disk
		static std::map<std::string, long long> s_PathModifyCache;
	};

	template<class ObjectType>
	inline ObjectType* AssetDB::CreateAssetObject(const Guid& guid)
	{
		static_assert(std::is_base_of<Object, ObjectType>::value, "Type is not derived from Object.");
		ObjectType* object = Object::Create<ObjectType>();
		ObjectDB::AllocateIdToGuid(object, guid);
		return object;
	}

	// Use inside importers only
	template<class ObjectType>
	inline ObjectType* AssetDB::LoadAssetObject(const Guid& guid)
	{
		YamlSerializer serializer;
		std::filesystem::path dataPath = Path::GetAssetCachePath();
		serializer.Deserialize(dataPath.append(guid.ToString().append(".yaml")).string());
		auto& deserializedObjects = serializer.GetDeserializedObjects();
		if (deserializedObjects.size() > 0)
		{
			auto pair = deserializedObjects[0];
			Object* object = pair.first;
			ObjectDB::AllocateIdToGuid(object, guid, pair.second);
			object->Initialize();
			return (ObjectType*)object;
		}
		return nullptr;
	}

	#define REGISTER_ASSET_IMPORTER( fileExtension, importerType ) AssetDB::Register(fileExtension, importerType);
}