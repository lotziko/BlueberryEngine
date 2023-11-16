#pragma once
#include <filesystem>

#include "Blueberry\Core\Guid.h"

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

		template<class ObjectType, typename... Args>
		static Ref<ObjectType> CreateAssetObject(const Guid& guid, Args&&... params);

	private:
		static void Import(const std::filesystem::path& path);

	public:
		static void Register(const std::string& extension, const std::size_t& importerType);

	private:
		static std::map<std::string, long long> s_PathModifyCache;
		static std::map<std::string, std::size_t> s_ImporterTypes;
		static std::map<Guid, Ref<AssetImporter>> s_Importers;
		static std::map<Guid, Ref<Object>> s_ImportedObjects;
	};

	template<class ObjectType, typename... Args>
	inline Ref<ObjectType> AssetDB::CreateAssetObject(const Guid& guid, Args&&... params)
	{
		static_assert(std::is_base_of<Object, ObjectType>::value, "Type is not derived from Object.");
		Ref<ObjectType> object = ObjectDB::CreateGuidObject<ObjectType>(guid, std::forward<Args>(params)...);
		s_ImportedObjects.insert_or_assign(guid, std::dynamic_pointer_cast<Object>(object));
		return object;
	}

	#define REGISTER_ASSET_IMPORTER( fileExtension, importerType ) AssetDB::Register(fileExtension, importerType);
}