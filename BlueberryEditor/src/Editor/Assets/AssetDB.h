#pragma once
#include <filesystem>

#include "Editor\Path.h"
#include "Editor\Serialization\YamlSerializer.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class AssetImporter;

	using AssetDBRefreshEvent = Event<>;

	class AssetDB
	{
	public:
		static void Refresh();

		static AssetImporter* GetImporter(const std::string& path);
		static AssetImporter* GetImporter(const Guid& guid);

		template<class ObjectType>
		static ObjectType* CreateAssetObject(const Guid& guid);

		// Use inside importers only
		static std::vector<std::pair<Object*, FileId>> AssetDB::LoadAssetObjects(const Guid& guid, const std::unordered_map<FileId, ObjectId>& existingObjects);

		static const std::string GetRelativeAssetPath(Object* object);
		static std::string GetAssetCachedDataPath(Object* object);

		static bool HasAssetWithGuidInData(const Guid& guid);
		static void CreateAsset(Object* object, const std::string& relativePath);
		static void SaveAssetObjectsToCache(const std::vector<Object*>& objects);
		static void SetDirty(Object* object);
		static void DeleteAssetFromData(const Guid& guid);
		static void SaveAssets();

		static AssetDBRefreshEvent& GetAssetDBRefreshed();

	private:
		static AssetImporter* CreateOrGetImporter(const std::filesystem::path& path);

	public:
		static void Register(const std::string& extension, const std::size_t& importerType);

	private:
		static std::unordered_map<std::string, std::size_t> s_ImporterTypes;
		static std::unordered_map<std::string, AssetImporter*> s_Importers;
		static std::unordered_map<Guid, std::string> s_GuidToPath;
		static std::vector<ObjectId> s_DirtyAssets;
		static AssetDBRefreshEvent s_AssetDBRefreshed;
	};

	template<class ObjectType>
	inline ObjectType* AssetDB::CreateAssetObject(const Guid& guid)
	{
		static_assert(std::is_base_of<Object, ObjectType>::value, "Type is not derived from Object.");
		ObjectType* object = Object::Create<ObjectType>();
		ObjectDB::AllocateIdToGuid(object, guid);
		return object;
	}

	#define REGISTER_ASSET_IMPORTER( fileExtension, importerType ) AssetDB::Register(fileExtension, importerType);
}