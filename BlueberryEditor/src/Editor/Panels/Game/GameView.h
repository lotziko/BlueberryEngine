#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class RenderTexture;

	class GameView : public EditorWindow
	{
		OBJECT_DECLARATION(GameView)

	public:
		GameView() = default;
		virtual ~GameView() = default;

		static void Open();

		virtual void OnDrawUI();

	private:
		RenderTexture* m_RenderTarget = nullptr;
	};
}