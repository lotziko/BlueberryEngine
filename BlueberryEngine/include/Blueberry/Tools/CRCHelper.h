#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class CRCHelper
	{
	public:
		static constexpr uint32_t Calculate(const void* data, size_t length, uint32_t previousCrc32 = 0)
		{
			// Based on https://create.stephan-brumme.com/crc32/
			const uint32_t polynomial = 0xEDB88320;
			uint32_t crc = ~previousCrc32;
			unsigned char* current = const_cast<unsigned char*>(static_cast<const unsigned char*>(data));
			while (length--)
			{
				crc ^= *current++;
				for (unsigned int j = 0; j < 8; j++)
				{
					crc = (crc >> 1) ^ (-int(crc & 1) & polynomial);
				}
			}
			return ~crc; // same as crc ^ 0xFFFFFFFF
		}

		static constexpr uint32_t Calculate(uint32_t data, uint32_t previousCrc32 = 0)
		{
			const uint32_t polynomial = 0xEDB88320;
			size_t length = sizeof(uint32_t);
			uint32_t crc = ~previousCrc32;
			while (length--)
			{
				crc ^= data++;
				for (unsigned int j = 0; j < 8; j++)
				{
					crc = (crc >> 1) ^ (-int(crc & 1) & polynomial);
				}
			}
			return ~crc; // same as crc ^ 0xFFFFFFFF
		}
	};
}