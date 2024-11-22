#pragma once

namespace Blueberry
{
	class RenderTexture;

	class GameView
	{
	public:
		void DrawUI();

	private:
		RenderTexture* m_RenderTarget = nullptr;
	};
}