#pragma once

namespace Blueberry
{
	class GfxShader
	{
	public:
		virtual ~GfxShader() = default;

	protected:
		std::unordered_map<size_t, uint32_t> m_ConstantBufferSlots;
		std::unordered_map<size_t, std::pair<uint32_t, uint32_t>> m_StructuredBufferSlots;
		std::unordered_map<size_t, std::pair<uint32_t, uint32_t>> m_TextureSlots;

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