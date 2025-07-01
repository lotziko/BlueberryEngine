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

		static AssetImporter* GetImporter(const String& relativePath);
		static AssetImporter* GetImporter(const Guid& guid);

		template<class ObjectType>
		static ObjectType* CreateAssetObject(const Guid& guid);

		// Use inside importers only
		static List<std::pair<Object*, FileId>> AssetDB::LoadAssetObjects(const Guid& guid, const Dictionary<FileId, ObjectId>& existingObjects);

		static const String GetRelativeAssetPath(Object* object);
		static String GetAssetCachedDataPath(Object* object);

		static bool HasAssetWithGuidInData(const Guid& guid);
		static void CreateAsset(Object* object, const String& relativePath);
		static void SaveAssetObjectsToCache(const List<Object*>& objects);
		static void SetDirty(Object* object);
		static void DeleteAssetFromData(const Guid& guid);
		static void SaveAssets();

		static AssetDBRefreshEvent& GetAssetDBRefreshed();

	private:
		static AssetImporter* CreateOrGetImporter(const std::filesystem::path& path);

	public:
		static void Register(const String& extension, const size_t& importerType);

	private:
		static Dictionary<String, size_t> s_ImporterTypes;
		static Dictionary<String, AssetImporter*> s_Importers;
		static Dictionary<Guid, String> s_GuidToPath;
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