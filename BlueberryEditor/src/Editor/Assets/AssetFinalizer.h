#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class Object;

	class AssetFinalizer
	{
	public:
		static void Finalize(Object* object, const Guid& guid, const FileId& fileId);
	};
}