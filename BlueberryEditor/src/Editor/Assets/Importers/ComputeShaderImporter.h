#pragma once

#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class ComputeShaderImporter : public AssetImporter
	{
		OBJECT_DECLARATION(ComputeShaderImporter)

	public:
		ComputeShaderImporter() = default;

	protected:
		virtual void ImportData() override;

	private:
		String GetShaderFolder();
	};
}