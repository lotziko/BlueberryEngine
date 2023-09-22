#pragma once

namespace Blueberry
{
	class Mesh;
	class Material;
	class GfxShader;
	class GfxTexture;
	class GfxVertexBuffer;
	class GfxIndexBuffer;

	struct GfxDrawingOperation
	{
		GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount);

		GfxShader* shader;
		std::vector<std::pair<std::size_t, GfxTexture*>>* textures;
		GfxVertexBuffer* vertexBuffer;
		GfxIndexBuffer* indexBuffer;
		UINT indexCount;
	};
}