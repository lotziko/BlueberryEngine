#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class SearchWindow : public EditorWindow
	{
		OBJECT_DECLARATION(SearchWindow)

	public:
		SearchWindow() = default;
		virtual ~SearchWindow() = default;

		static void Open(const Vector2& position);

		virtual void OnDrawUI() final;

	private:
		Vector2 m_Position;
	};
}