#pragma once

#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class ComputeShaderImporter : public AssetImporter
	{
		OBJECT_DECLARATION(ComputeShaderImporter)

	public:
		ComputeShaderImporter() = default;

		static String GetShaderFolder(const Guid& guid);

	protected:
		virtual void ImportData() override;
	};
}