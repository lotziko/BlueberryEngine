#pragma once

namespace Blueberry
{
	class Mesh;
	class Material;
	class GfxVertexShader;
	class GfxGeometryShader;
	class GfxFragmentShader;
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

		bool isValid = true;
		GfxVertexShader* vertexShader;
		GfxGeometryShader* geometryShader;
		GfxFragmentShader* fragmentShader;
		std::pair<size_t, GfxTexture*> textures[16] = {};
		UINT textureCount;
		GfxVertexBuffer* vertexBuffer;
		GfxIndexBuffer* indexBuffer;
		CullMode cullMode;
		BlendMode blendSrc;
		BlendMode blendDst;
		ZWrite zWrite;
		Topology topology;
		UINT indexCount;
		UINT indexOffset;
		ObjectId materialId;
	};
}