#include "bbpch.h"
#include "Screen.h"

namespace Blueberry
{
	uint32_t Screen::s_Width = 0;
	uint32_t Screen::s_Height = 0;
	float Screen::s_Scale = 1.0f;

	const uint32_t& Screen::GetWidth()
	{
		return s_Width;
	}

	const uint32_t& Screen::GetHeight()
	{
		return s_Height;
	}

	const float& Screen::GetScale()
	{
		return s_Scale;
	}
}
