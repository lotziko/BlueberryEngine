#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class FileHelper
	{
	public:
		static void Save(const uint8_t* data, const size_t& length, const String& path);
		static void Load(uint8_t*& data, size_t& length, const String& path);
		static void Load(String& data, const String& path);
	};
}