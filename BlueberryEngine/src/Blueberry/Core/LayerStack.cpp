#include "bbpch.h"
#include "LayerStack.h"

#include "Layer.h"

namespace Blueberry
{
	LayerStack::~LayerStack()
	{
		for (Layer* layer : m_Layers)
		{
			layer->OnDetach();
			delete layer;
		}
	}

	void LayerStack::PushLayer(Layer* layer)
	{
		m_Layers.emplace_back(layer);
	}

	void LayerStack::PopLayer(Layer* layer)
	{
		auto it = std::find(m_Layers.begin(), m_Layers.end(), layer);
		if (it != m_Layers.end())
		{
			layer->OnDetach();
			m_Layers.erase(it);
		}
	}

	std::vector<Layer*>::iterator LayerStack::begin()
	{
		return m_Layers.begin();
	}

	std::vector<Layer*>::iterator LayerStack::end()
	{
		return m_Layers.end();
	}
}