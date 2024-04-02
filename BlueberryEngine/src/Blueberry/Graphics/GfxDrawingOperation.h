#pragma once

namespace Blueberry
{
	class Mesh;
	class Material;
	class GfxShader;
	class GfxTexture;
	class GfxVertexBuffer;
	class GfxIndexBuffer;
	enum class CullMode;
	enum class BlendMode;
	enum class ZWrite;
	enum class Topology;

	struct GfxDrawingOperation
	{
		GfxDrawingOperation(GfxVertexBuffer* vertexBuffer, GfxIndexBuffer* indexBuffer, Material* material, const UINT& indexCount, const UINT& indexOffset, const Topology& topology);
		GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount, const UINT& indexOffset);
		GfxDrawingOperation(Mesh* mesh, Material* material);

		bool IsValid() const;

		GfxShader* shader;
		std::vector<std::pair<std::size_t, GfxTexture*>>* textures;
		GfxVertexBuffer* vertexBuffer;
		GfxIndexBuffer* indexBuffer;
		CullMode cullMode;
		BlendMode blendSrc;
		BlendMode blendDst;
		ZWrite zWrite;
		Topology topology;
		UINT indexCount;
		UINT indexOffset;
	};
}