#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class GfxBottomLevelAccelerationStructure;
	class Material;

	class GfxTopLevelAccelerationStructure
	{
	public:
		virtual void Add(GfxBottomLevelAccelerationStructure* accelerationStructure, const List<ObjectPtr<Material>>& materials, const Matrix& transform) = 0;
		virtual void Clear() = 0;
	};
}