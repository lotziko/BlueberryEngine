#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class FileHelper
	{
	public:
		static void Save(const uint8_t* data, size_t length, const String& path);
		static void Save(const List<uint8_t>& data, const String& path);
		static List<uint8_t> LoadBinary(const String& path);
		static String LoadText(const String& path);
	};
}