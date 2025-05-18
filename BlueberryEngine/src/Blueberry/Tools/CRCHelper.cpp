#include "Blueberry\Tools\CRCHelper.h"

namespace Blueberry
{
	const uint32_t Polynomial = 0xEDB88320;

	// Based on https://create.stephan-brumme.com/crc32/
	uint32_t CRCHelper::Calculate(const void* data, size_t length, uint32_t previousCrc32)
	{
		uint32_t crc = ~previousCrc32;
		unsigned char* current = const_cast<unsigned char*>(static_cast<const unsigned char*>(data));
		while (length--)
		{
			crc ^= *current++;
			for (unsigned int j = 0; j < 8; j++)
			{
				crc = (crc >> 1) ^ (-int(crc & 1) & Polynomial);
			}
		}
		return ~crc; // same as crc ^ 0xFFFFFFFF
	}

	uint32_t CRCHelper::Calculate(uint32_t data, uint32_t previousCrc32)
	{
		size_t length = sizeof(uint32_t);
		uint32_t crc = ~previousCrc32;
		while (length--)
		{
			crc ^= data++;
			for (unsigned int j = 0; j < 8; j++)
			{
				crc = (crc >> 1) ^ (-int(crc & 1) & Polynomial);
			}
		}
		return ~crc; // same as crc ^ 0xFFFFFFFF
	}
}
