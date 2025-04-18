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

		const Guid& GetGuid();
		std::string GetFilePath();
		std::string GetMetaFilePath();
		const std::string& GetRelativeFilePath();
		const std::string& GetRelativeMetaFilePath();
		const FileId& GetMainObject();
		const Dictionary<FileId, ObjectId>& GetAssetObjects();
		const bool IsImported();
		const bool& IsRequiringSave();

		void ResetImport();
		void ImportDataIfNeeded();
		void Save();
		// TODO need a way to determine count of not imported assets in this importer
		
		static AssetImporter* CreateNew(const size_t& type, const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath);
		static AssetImporter* CreateFromMeta(const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath);
		static void LoadFromMeta(AssetImporter* importer);
		
	protected:
		virtual void ImportData() = 0;
		void AddAssetObject(Object* object, const FileId& fileId);
		void SetMainObject(const FileId& id);

	private:
		Guid m_Guid;
		std::string m_RelativePath;
		std::string m_RelativeMetaPath;
		FileId m_MainObject;
		Dictionary<FileId, ObjectId> m_AssetObjects = {};
		//Texture2D* m_Icon = nullptr;
		bool m_RequireSave = false;

		friend class ImporterInfoCache;
	};
}