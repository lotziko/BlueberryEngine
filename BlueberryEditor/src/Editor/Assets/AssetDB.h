#pragma once
#include <filesystem>

#include "Editor\Path.h"
#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Events\Event.h"

namespace Blueberry
{
	class AssetImporter;
	class ObjectFinalizer;
	class Serializer;

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
		static const String GetRelativePath(const Guid& guid);
		static String GetAssetCachedDataPath(Object* object);

		static void GetDependent(const Guid& guid, HashSet<Guid>& dependent);
		static void SetDependencies(const Guid& guid, const HashSet<Guid>& dependencies);

		static bool HasAssetWithGuidInData(const Guid& guid);
		static void CreateAsset(Object* object, const String& relativePath);
		static void SaveAssetObjectsToCache(const List<Object*>& objects);
		static void SetDirty(Object* object);
		static void DeleteAsset(Object* object);
		static void DeleteAssetFromData(const Guid& guid);
		static void MarkForReimport(const Guid& guid);
		static void SaveAssets();

		static AssetDBRefreshEvent& GetAssetDBRefreshed();

	private:
		static AssetImporter* CreateOrGetImporter(const std::filesystem::path& path);
		static AssetImporter* CreateImporter(const std::filesystem::path& path);

	public:
		static void Register(const String& extension, const TypeId& importerType);

	private:
		static Dictionary<String, TypeId> s_ImporterTypes;
		static Dictionary<String, AssetImporter*> s_Importers;
		static List<std::pair<TypeId, ObjectFinalizer*>> s_Finalizers;
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

	#define REGISTER_ASSET_IMPORTER( fileExtension, importerType ) AssetDB::Register(fileExtension, importerType)
}