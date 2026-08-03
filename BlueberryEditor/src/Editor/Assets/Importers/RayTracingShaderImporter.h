#pragma once

#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class RayTracingShaderImporter : public AssetImporter
	{
		OBJECT_DECLARATION(RayTracingShaderImporter)

	public:
		RayTracingShaderImporter() = default;

		static String GetShaderFolder(const Guid& guid);

	protected:
		virtual bool IsRequiringReimport() const final;
		virtual void ImportData() final;
	};
}