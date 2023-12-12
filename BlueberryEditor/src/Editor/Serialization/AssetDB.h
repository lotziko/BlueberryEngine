#pragma once
#include <filesystem>

#include "Editor\Path.h"
#include "Editor\Serialization\YamlSerializer.h"
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

		template<class ObjectType>
		static Ref<ObjectType> LoadAssetObject(const Guid& guid);

		static std::string GetAssetDataPath(Object* object, const char* extension);

		static bool HasAssetWithGuidInData(const Guid& guid);
		static void SaveAssetObject(Object* object);

	private:
		static void Import(const std::filesystem::path& path);

	public:
		static void Register(const std::string& extension, const std::size_t& importerType);

	private:
		static std::map<std::string, long long> s_PathModifyCache;
		static std::map<std::string, std::size_t> s_ImporterTypes;
		static std::map<Guid, Ref<AssetImporter>> s_Importers;
	};

	template<class ObjectType, typename... Args>
	inline Ref<ObjectType> AssetDB::CreateAssetObject(const Guid& guid, Args&&... params)
	{
		static_assert(std::is_base_of<Object, ObjectType>::value, "Type is not derived from Object.");
		Ref<ObjectType> object = ObjectDB::CreateObject<ObjectType>(std::forward<Args>(params)...);
		ObjectDB::AddObjectGuid(object->GetObjectId(), guid);
		return object;
	}

	template<class ObjectType>
	inline Ref<ObjectType> AssetDB::LoadAssetObject(const Guid& guid)
	{
		YamlSerializer serializer;
		std::filesystem::path dataPath = Path::GetDataPath();
		serializer.Deserialize(dataPath.append(guid.ToString().append(".yaml")).string());
		auto& deserializedObjects = serializer.GetDeserializedObjects();
		if (deserializedObjects.size() > 0)
		{
			ObjectDB::AddObjectGuid(deserializedObjects[0]->GetObjectId(), guid);
			return std::dynamic_pointer_cast<ObjectType>(deserializedObjects[0]);
		}
		return nullptr;
	}

	#define REGISTER_ASSET_IMPORTER( fileExtension, importerType ) AssetDB::Register(fileExtension, importerType);
}