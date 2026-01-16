#pragma once

#include "Editor\Assets\ObjectFinalizer.h"

namespace Blueberry
{
	class ComputeShaderFinalizer : public ObjectFinalizer
	{
	public:
		virtual void Finalize(Object* object, const Guid& guid, const FileId& fileId) final;
	};
}