#pragma once

namespace Blueberry
{
	class Object;
	class Texture2D;
	class RenderTexture;

	class ThumbnailRenderer
	{
	public:
		static bool CanDraw(const size_t& type);
		static bool Draw(unsigned char* output, const UINT& size, Object* asset);

	private:
		static inline RenderTexture* s_ThumbnailRenderTarget = nullptr;
	};
}