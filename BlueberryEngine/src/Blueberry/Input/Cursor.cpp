#include "Blueberry\Input\Cursor.h"

namespace Blueberry
{
	bool Cursor::s_Locked = false;
	bool Cursor::s_Hidden = false;

	const bool& Cursor::IsLocked()
	{
		return s_Locked;
	}

	void Cursor::SetLocked(const bool& locked)
	{
		s_Locked = locked;
	}

	const bool& Cursor::IsHidden()
	{
		return s_Hidden;
	}

	void Cursor::SetHidden(const bool& hidden)
	{
		s_Hidden = hidden;
	}
}
