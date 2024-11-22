#pragma once

namespace Blueberry
{
	class Object;
	class Texture2D;
	class RenderTexture;

	class ThumbnailRenderer
	{
	public:
		static bool CanDraw(const std::size_t& type);
		static bool Draw(unsigned char* output, const uint32_t& size, Object* asset);

	private:
		static inline RenderTexture* s_ThumbnailRenderTarget = nullptr;
	};
}