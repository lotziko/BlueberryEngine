#pragma once

namespace Blueberry
{
	class Texture2D;

	class DefaultTextures
	{
	public:
		static Texture2D* GetTexture(const std::string& name);
		static Texture2D* GetWhite();
		static Texture2D* GetNormal();

	private:
		static inline Texture2D* s_WhiteTexture = nullptr;
		static inline Texture2D* s_NormalTexture = nullptr;
	};
}