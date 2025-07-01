#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Texture;
	class Texture2D;
	class TextureCube;
	class Texture3D;
	enum class TextureDimension;

	class DefaultTextures
	{
	public:
		static Texture* GetTexture(const String& name, const TextureDimension& dimension);
		static Texture2D* GetWhite2D();
		static Texture2D* GetBlack2D();
		static Texture2D* GetNormal2D();
		static TextureCube* GetWhiteCube();
		static TextureCube* GetBlackCube();
		static Texture3D* GetWhite3D();
		static Texture3D* GetBlack3D();

	private:
		static inline Texture2D* s_WhiteTexture2D = nullptr;
		static inline Texture2D* s_BlackTexture2D = nullptr;
		static inline Texture2D* s_NormalTexture2D = nullptr;
		static inline TextureCube* s_WhiteTextureCube = nullptr;
		static inline TextureCube* s_BlackTextureCube = nullptr;
		static inline Texture3D* s_WhiteTexture3D = nullptr;
		static inline Texture3D* s_BlackTexture3D = nullptr;
	};
}