#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class BB_API Cursor
	{
	public:
		static const bool& IsLocked();
		static void SetLocked(const bool& locked);

		static const bool& IsHidden();
		static void SetHidden(const bool& hidden);

	private:
		static inline bool s_Locked = false;
		static inline bool s_Hidden = false;
	};
}