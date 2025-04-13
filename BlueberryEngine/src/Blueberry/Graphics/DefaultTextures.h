#pragma once

namespace Blueberry
{
	class Texture;
	class Texture2D;
	class TextureCube;
	enum class TextureDimension;

	class DefaultTextures
	{
	public:
		static Texture* GetTexture(const std::string& name, const TextureDimension& dimension);
		static Texture2D* GetWhite2D();
		static Texture2D* GetNormal2D();
		static TextureCube* GetWhiteCube();

	private:
		static inline Texture2D* s_WhiteTexture2D = nullptr;
		static inline Texture2D* s_NormalTexture2D = nullptr;
		static inline TextureCube* s_WhiteTextureCube = nullptr;
	};
}