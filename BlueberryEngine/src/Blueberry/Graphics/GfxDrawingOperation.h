#pragma once

namespace Blueberry
{
	class Mesh;
	class Material;
	class GfxShader;
	class GfxTexture;
	class GfxVertexBuffer;
	class GfxIndexBuffer;
	enum Topology;

	struct GfxDrawingOperation
	{
		GfxDrawingOperation(GfxVertexBuffer* vertexBuffer, GfxIndexBuffer* indexBuffer, Material* material, const UINT& indexCount, const Topology& topology);
		GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount);
		GfxDrawingOperation(Mesh* mesh, Material* material);

		bool IsValid() const;

		GfxShader* shader;
		std::vector<std::pair<std::size_t, GfxTexture*>>* textures;
		GfxVertexBuffer* vertexBuffer;
		GfxIndexBuffer* indexBuffer;
		Topology topology;
		UINT indexCount;
	};
}