#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class StatisticsWindow : public EditorWindow
	{
		OBJECT_DECLARATION(StatisticsWindow)

	public:
		StatisticsWindow() = default;
		virtual ~StatisticsWindow() = default;

		static void Open();

		virtual void OnDrawUI() final;
	};
}