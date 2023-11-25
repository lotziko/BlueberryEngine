#pragma once
#include <filesystem>

#include "Editor\Path.h"
#include "Blueberry\Core\Guid.h"
#include "Blueberry\Serialization\YamlHelper.h"

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

		template<class ObjectType>
		static Ref<ObjectType> LoadAssetObject(const Guid& guid);

		static bool HasAssetWithGuidInData(const Guid& guid);
		static void SaveAssetObject(Ref<Object> object);

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

	template<class ObjectType>
	inline Ref<ObjectType> AssetDB::LoadAssetObject(const Guid& guid)
	{
		auto dataPath = Path::GetDataPath();
		dataPath.append(guid.ToString().append(".yaml"));
		ryml::Tree tree;
		YamlHelper::Load(tree, dataPath.string());
		Ref<ObjectType> object = ObjectDB::CreateGuidObject<ObjectType>(guid);
		object->Deserialize(tree.rootref());
		return object;
	}

	#define REGISTER_ASSET_IMPORTER( fileExtension, importerType ) AssetDB::Register(fileExtension, importerType);
}