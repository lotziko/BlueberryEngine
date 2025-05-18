#include "Layer.h"

namespace Blueberry
{
	Layer::Layer(const String& name) : m_Name(name)
	{
	}

	const String& Layer::GetName() const
	{
		return m_Name;
	}
}