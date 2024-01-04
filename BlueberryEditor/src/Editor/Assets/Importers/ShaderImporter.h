#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class ShaderImporter : public AssetImporter
	{
		OBJECT_DECLARATION(ShaderImporter)

	public:
		ShaderImporter() = default;

		static void BindProperties();

	protected:
		virtual void ImportData() override;

	private:
		std::string GetShaderPath(const char* extension);
	};
}