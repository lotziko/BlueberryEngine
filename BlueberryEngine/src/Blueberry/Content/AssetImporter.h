#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include <map>

namespace Blueberry
{
	class AssetImporter
	{
	public:
		virtual Ref<Object> Import(const std::string& path) = 0;
		virtual std::size_t GetType() = 0;
	};
}