#pragma once

namespace Blueberry
{
	class CRCHelper
	{
	public:
		static uint32_t Calculate(const void* data, size_t length, uint32_t previousCrc32 = 0);
		static uint32_t Calculate(uint32_t data, uint32_t previousCrc32 = 0);
	};
}