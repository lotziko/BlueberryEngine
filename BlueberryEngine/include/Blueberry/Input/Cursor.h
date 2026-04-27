#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class BB_API Cursor
	{
	public:
		static bool IsLocked();
		static void SetLocked(bool locked);

		static bool IsHidden();
		static void SetHidden(bool hidden);

	private:
		static bool s_Locked;
		static bool s_Hidden;
	};
}