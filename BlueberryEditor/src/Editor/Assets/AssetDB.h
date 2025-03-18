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
		static List<std::pair<Object*, FileId>> AssetDB::LoadAssetObjects(const Guid& guid, const Dictionary<FileId, ObjectId>& existingObjects);

		static const std::string GetRelativeAssetPath(Object* object);
		static std::string GetAssetCachedDataPath(Object* object);

		static bool HasAssetWithGuidInData(const Guid& guid);
		static void CreateAsset(Object* object, const std::string& relativePath);
		static void SaveAssetObjectsToCache(const List<Object*>& objects);
		static void SetDirty(Object* object);
		static void DeleteAssetFromData(const Guid& guid);
		static void SaveAssets();

		static AssetDBRefreshEvent& GetAssetDBRefreshed();

	private:
		static AssetImporter* CreateOrGetImporter(const std::filesystem::path& path);

	public:
		static void Register(const std::string& extension, const std::size_t& importerType);

	private:
		static Dictionary<std::string, std::size_t> s_ImporterTypes;
		static Dictionary<std::string, AssetImporter*> s_Importers;
		static Dictionary<Guid, std::string> s_GuidToPath;
		static List<ObjectId> s_DirtyAssets;
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