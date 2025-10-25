#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class ShaderImporter : public AssetImporter
	{
		OBJECT_DECLARATION(ShaderImporter)

	public:
		ShaderImporter() = default;

		static String GetShaderFolder(const Guid& guid);
		static long long GetLastFilesWriteTime();

	protected:
		virtual void ImportData() override;
	};
}