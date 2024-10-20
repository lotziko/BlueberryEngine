#include "bbpch.h"
#include "Screen.h"

namespace Blueberry
{
	UINT Screen::s_Width = 0;
	UINT Screen::s_Height = 0;

	const UINT& Screen::GetWidth()
	{
		return s_Width;
	}

	const UINT& Screen::GetHeight()
	{
		return s_Height;
	}
}
