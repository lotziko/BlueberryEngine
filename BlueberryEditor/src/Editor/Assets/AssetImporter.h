#pragma once
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Guid.h"
#include <filesystem>

namespace Blueberry
{
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
		const std::map<FileId, ObjectId>& GetImportedObjects();
		
		void ImportDataIfNeeded();
		void Save();
		// TODO need a way to determine count of not imported assets in this importer
		
		static AssetImporter* Create(const size_t& type, const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath);
		static AssetImporter* Load(const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath);

		static void BindProperties();
		
	protected:
		virtual void ImportData() = 0;
		void AddImportedObject(Object* object, const FileId& fileId);

	private:
		Guid m_Guid;
		std::string m_RelativePath;
		std::string m_RelativeMetaPath;
		std::map<FileId, ObjectId> m_ImportedObjects;
	};
}