#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class CompressionHelper
	{
	public:
		static void Decompress(const List<uint8_t>& data, List<uint8_t>& decompressedData);
		static List<uint8_t> Decompress(const List<uint8_t>& data, size_t originalSize);
		static List<uint8_t> Compress(const List<uint8_t>& data);
		static List<uint8_t> Compress(const String& data);
	};
}