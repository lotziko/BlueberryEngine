#pragma once

namespace Blueberry
{
	class GfxShader
	{
	public:
		virtual ~GfxShader() = default;

	protected:
		std::unordered_map<std::size_t, UINT> m_VertexConstantBufferSlots;
		std::unordered_map<std::size_t, UINT> m_PixelConstantBufferSlots;
		std::unordered_map<std::size_t, UINT> m_TextureSlots;

		friend struct GfxDrawingOperation;
	};
}