#pragma once

#include "Blueberry\Core\Guid.h"
#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class AssetImporter;

	class ImporterInfoCache
	{
	private:
		struct ImporterInfo
		{
			void Read(std::istream& is);
			void Write(std::ostream& os) const;

			FileId mainObject;
			List<std::tuple<FileId, TypeId, String>> objects;
		};

	public:
		static void Load();
		static void Save();

		static bool Has(AssetImporter* importer);
		static bool Get(AssetImporter* importer);
		static void Set(AssetImporter* importer);

	private:
		static Dictionary<Guid, ImporterInfo> s_ImporterInfoCache;
	};
}