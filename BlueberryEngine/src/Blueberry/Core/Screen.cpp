#include "bbpch.h"
#include "Screen.h"

namespace Blueberry
{
	uint32_t Screen::s_Width = 0;
	uint32_t Screen::s_Height = 0;

	const uint32_t& Screen::GetWidth()
	{
		return s_Width;
	}

	const uint32_t& Screen::GetHeight()
	{
		return s_Height;
	}
}
