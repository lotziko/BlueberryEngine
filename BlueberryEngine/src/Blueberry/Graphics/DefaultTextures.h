#pragma once

namespace Blueberry
{
	class Texture2D;

	class DefaultTextures
	{
	public:
		static Texture2D* GetWhite();

	private:
		static inline Texture2D* s_WhiteTexture = nullptr;
	};
}