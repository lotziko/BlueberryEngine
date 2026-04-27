#include "Blueberry\Input\Cursor.h"

namespace Blueberry
{
	bool Cursor::s_Locked = false;
	bool Cursor::s_Hidden = false;

	bool Cursor::IsLocked()
	{
		return s_Locked;
	}

	void Cursor::SetLocked(bool locked)
	{
		s_Locked = locked;
	}

	bool Cursor::IsHidden()
	{
		return s_Hidden;
	}

	void Cursor::SetHidden(bool hidden)
	{
		s_Hidden = hidden;
	}
}
