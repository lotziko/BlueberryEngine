#include "bbpch.h"
#include "Variant.h"

namespace Blueberry
{
	Variant::Variant(void* data) : m_Data(data)
	{
	}

	Variant::Variant(void* data, const uint32_t& offset)
	{
		m_Data = reinterpret_cast<char*>(data) + offset;
	}
}
