#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class Object;

	class ObjectFinalizer
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual void Finalize(Object* object, const Guid& guid, const FileId& fileId) = 0;
	};
}