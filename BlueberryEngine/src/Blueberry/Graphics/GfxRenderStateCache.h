#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Material;

	class GfxVertexShader;
	class GfxGeometryShader;
	class GfxFragmentShader;

	enum class CullMode;
	enum class BlendMode;
	enum class ZTest;
	enum class ZWrite;

	struct GfxPassData
	{
		GfxVertexShader* vertexShader;
		GfxGeometryShader* geometryShader;
		GfxFragmentShader* fragmentShader;

		CullMode cullMode;
		BlendMode blendSrcColor;
		BlendMode blendSrcAlpha;
		BlendMode blendDstColor;
		BlendMode blendDstAlpha;
		ZTest zTest;
		ZWrite zWrite;
		bool isValid;
	};

	class GfxRenderStateCache
	{
	protected:
		GfxPassData GetPassData(Material* material, const uint8_t& passIndex);
		uint32_t GetTextureIndex(Material* material, const size_t& id);
		uint32_t GetTextureIndex(Material* material, const uint32_t& slotIndex);
	};
}