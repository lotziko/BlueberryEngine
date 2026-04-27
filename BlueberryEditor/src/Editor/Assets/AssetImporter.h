#pragma once
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Guid.h"
#include <filesystem>

namespace Blueberry
{
	class Texture2D;

	class AssetImporter : public Object
	{
		OBJECT_DECLARATION(AssetImporter)

	public:
		AssetImporter() = default;

		const Guid& GetGuid() const;
		String GetFilePath();
		String GetMetaFilePath();
		const String& GetRelativeFilePath();
		const String& GetRelativeMetaFilePath();
		FileId GetMainObject() const;
		const Dictionary<FileId, ObjectId>& GetAssetObjects();
		bool IsImported();
		virtual bool IsRequiringReimport() const;
		bool IsRequiringSave() const;

		void ResetImport();
		void ImportDataIfNeeded();
		void Save();
		void SaveAndReimport();
		void Rename(const std::filesystem::path& relativePath);
		long long GetLastWrite() const;
		// TODO need a way to determine count of not imported assets in this importer
		
		static AssetImporter* CreateNew(const TypeId& type, const std::filesystem::path& relativePath);
		static AssetImporter* CreateFromMeta(const std::filesystem::path& relativePath);
		static void LoadFromMeta(AssetImporter* importer);
		
	protected:
		virtual bool IsImportable() const;
		virtual void ImportData() = 0;
		void LoadData();
		void AddAssetObject(Object* object, const FileId& fileId);
		void SetMainObject(const FileId& id);
		template<class ObjectType>
		ObjectType* GetOrCreateAssetObject(const FileId& fileId);

	private:
		Guid m_Guid;
		String m_RelativePath;
		String m_RelativeMetaPath;
		FileId m_MainObject;
		Dictionary<FileId, ObjectId> m_AssetObjects = {};
		bool m_RequireSave = false;
		long long m_LastWrite = 0;

		friend class ImporterInfoCache;
	};

	template<class ObjectType>
	inline ObjectType* AssetImporter::GetOrCreateAssetObject(const FileId& fileId)
	{
		Guid guid = GetGuid();
		ObjectType* result;
		auto& objects = ObjectDB::GetObjectsFromGuid(guid);
		auto it = objects.find(fileId);
		if (it != objects.end())
		{
			Object* object = ObjectDB::GetObject(it->second);
			if (object != nullptr)
			{
				result = static_cast<ObjectType*>(object);
				result->SetState(ObjectState::Default);
			}
			else
			{
				result = Object::Create<ObjectType>();
			}
		}
		else
		{
			result = Object::Create<ObjectType>();
		}
		ObjectDB::AllocateIdToGuid(result, guid, fileId);
		return result;
	}
}