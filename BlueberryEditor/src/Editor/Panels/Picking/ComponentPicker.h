#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class ComponentPicker
	{
	public:
		// Build the cached category tree once for this opening.
		static void Open();
		// Only registered components return a TypeId; fake entries preview a selection.
		static bool GetResult(TypeId* type);
	};
}
