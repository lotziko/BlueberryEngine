#pragma once

namespace Blueberry
{
	class Layer;

	class LayerStack
	{
	public:
		BB_OVERRIDE_NEW_DELETE;

		LayerStack() = default;
		virtual ~LayerStack();

		void PushLayer(Layer* layer);
		void PopLayer(Layer* layer);

		List<Layer*>::iterator begin();
		List<Layer*>::iterator end();

	private:
		List<Layer*> m_Layers;
	};
}