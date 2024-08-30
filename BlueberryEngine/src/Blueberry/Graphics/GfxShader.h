#pragma once

namespace Blueberry
{
	class GfxShader
	{
	public:
		virtual ~GfxShader() = default;

	protected:
		std::unordered_map<std::size_t, UINT> m_ConstantBufferSlots;
		std::unordered_map<std::size_t, std::pair<UINT, UINT>> m_TextureSlots;

		friend struct GfxDrawingOperation;
	};
	
	class GfxVertexShader : public GfxShader
	{
	};

	class GfxGeometryShader : public GfxShader
	{
	};

	class GfxFragmentShader : public GfxShader
	{
	};
}