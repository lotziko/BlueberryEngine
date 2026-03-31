#pragma once

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Events\Event.h"

#include "Editor\Path.h"
#include "Editor\Misc\FileWatch.h"

#include <filesystem>

namespace Blueberry
{
	class AssetImporter;
	class ObjectFinalizer;
	class Serializer;

	using AssetDBRefreshEvent = Event<>;

	class AssetDB
	{
	public:
		static void Initialize();
		static void Shutdown();
		static void Refresh();

		static AssetImporter* GetImporter(const String& relativePath);
		static AssetImporter* GetImporter(const Guid& guid);

		template<class ObjectType>
		static ObjectType* CreateAssetObject(const Guid& guid);

		// Use inside importers only
		static List<std::pair<Object*, FileId>> AssetDB::LoadAssetObjects(const Guid& guid, const Dictionary<FileId, ObjectId>& existingObjects);

		static const String GetRelativeAssetPath(Object* object);
		static const String GetRelativePath(const Guid& guid);
		static String GetAssetCachedDataPath(Object* object);

		static void GetDependent(const Guid& guid, HashSet<Guid>& dependent);
		static void SetDependencies(const Guid& guid, const HashSet<Guid>& dependencies);

		static bool HasAssetWithGuidInData(const Guid& guid);
		static void CreateAsset(Object* object, const String& relativePath);
		static void SaveAssetObjectsToCache(const List<Object*>& objects);
		static void SetDirty(Object* object);
		static void ImportAsset(const String& relativePath);
		static void RenameAsset(const String& fromRelativePath, const String& toRelativePath);
		static void DeleteAsset(const String& relativePath);
		static void DeleteAssetFromData(const Guid& guid);
		static void SaveAssets();

		static AssetDBRefreshEvent& GetAssetDBRefreshed();

	private:
		static void InitializeImporters();
		static bool NeedsReimport(AssetImporter* importer);
		static void OnImportAsset(AssetImporter* importer);
		static void OnRenameAsset(AssetImporter* importer, const String& toRelativePath);
		static void OnDeleteAsset(AssetImporter* importer);
		static AssetImporter* CreateOrGetImporter(const std::filesystem::path& path);
		static AssetImporter* CreateImporter(const std::filesystem::path& path, const std::filesystem::path& relativePath);

	public:
		static void Register(const String& extension, const TypeId& importerType);

	private:
		static Dictionary<String, TypeId> s_ImporterTypes;
		static List<std::pair<TypeId, ObjectFinalizer*>> s_Finalizers;
		static List<ObjectId> s_DirtyAssets;
		static AssetDBRefreshEvent s_AssetDBRefreshed;
		static FileWatch* s_AssetsWatch;

		static Dictionary<String, ObjectId> s_RelativePathToImporterId;
		static Dictionary<Guid, ObjectId> s_GuidToImporterId;
		static Dictionary<Guid, String> s_GuidToRelativePath;
	};

	template<class ObjectType>
	inline ObjectType* AssetDB::CreateAssetObject(const Guid& guid)
	{
		static_assert(std::is_base_of<Object, ObjectType>::value, "Type is not derived from Object.");
		ObjectType* object = Object::Create<ObjectType>();
		ObjectDB::AllocateIdToGuid(object, guid);
		return object;
	}

	#define REGISTER_ASSET_IMPORTER( fileExtension, importerType ) AssetDB::Register(fileExtension, importerType)
}