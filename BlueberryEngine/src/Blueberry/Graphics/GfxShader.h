#pragma once

namespace Blueberry
{
	class GfxShader
	{
	public:
		virtual ~GfxShader() = default;

	protected:
		std::unordered_map<size_t, UINT> m_ConstantBufferSlots;
		std::unordered_map<size_t, std::pair<UINT, UINT>> m_StructuredBufferSlots;
		std::unordered_map<size_t, std::pair<UINT, UINT>> m_TextureSlots;

		friend struct GfxDrawingOperation;
		friend class Material;
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