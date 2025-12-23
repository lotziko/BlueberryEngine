#pragma once

#include "Blueberry\Core\Layer.h"

#include <functional>

namespace Blueberry
{
	class WindowResizeEventArgs;

	class HubLayer : public Layer
	{
	public:
		HubLayer(const std::function<void(Layer*, WString)>& callback);

		virtual void OnAttach() override;
		virtual void OnDetach() override;
		virtual void OnDraw() override;

		void OnWindowResize(const WindowResizeEventArgs& args);

	private:
		void DrawHub();

	private:
		std::function<void(Layer*, WString)> m_Callback;
	};
}