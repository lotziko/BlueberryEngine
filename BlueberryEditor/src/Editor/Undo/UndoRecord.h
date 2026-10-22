#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class UndoRecord
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual void OnUndo() = 0;
		virtual void OnRedo() = 0;
	};
}