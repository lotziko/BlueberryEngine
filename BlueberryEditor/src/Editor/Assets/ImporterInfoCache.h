#pragma once

#include "Blueberry\Core\Guid.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class AssetImporter;

	struct ImporterInfo
	{
		FileId mainObject;
		List<std::tuple<FileId, TypeId, String>> objects;
		long long lastWrite;
	};

	class ImporterInfoCache
	{
	public:
		static void Load();
		static void Save();

		static bool Has(AssetImporter* importer);
		static bool Get(AssetImporter* importer);
		static void Set(AssetImporter* importer);
		static void Clear(AssetImporter* importer);

	private:
		static void Read(std::istream& is, ImporterInfo& info);
		static void Write(std::ostream& os, ImporterInfo& info);

	private:
		static Dictionary<Guid, ImporterInfo> s_ImporterInfoCache;
	};
}