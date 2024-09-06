#pragma once

namespace Blueberry
{
	class Mesh;
	class Material;
	class GfxVertexShader;
	class GfxGeometryShader;
	class GfxFragmentShader;
	class GfxTexture;
	class Texture;
	class GfxVertexBuffer;
	class GfxIndexBuffer;
	enum class CullMode;
	enum class BlendMode;
	enum class ZTest;
	enum class ZWrite;
	enum class Topology;

	struct GfxRenderState
	{
		struct TextureInfo
		{
			GfxTexture* texture;
			UINT textureSlot;
			UINT samplerSlot;
		};

		GfxVertexShader* vertexShader;
		GfxGeometryShader* geometryShader;
		GfxFragmentShader* fragmentShader;

		TextureInfo fragmentTextures[16];
		UINT fragmentTextureCount;

		CullMode cullMode;
		BlendMode blendSrcColor;
		BlendMode blendSrcAlpha;
		BlendMode blendDstColor;
		BlendMode blendDstAlpha;
		ZTest zTest;
		ZWrite zWrite;

		bool isValid;
	};

	struct GfxDrawingOperation
	{
		GfxDrawingOperation(GfxVertexBuffer* vertexBuffer, GfxIndexBuffer* indexBuffer, Material* material, const UINT& indexCount, const UINT& indexOffset, const UINT& vertexCount, const Topology& topology);
		GfxDrawingOperation(Mesh* mesh, Material* material, const UINT& indexCount, const UINT& indexOffset, const UINT& vertexCount);
		GfxDrawingOperation(Mesh* mesh, Material* material);

		bool IsValid() const;

		bool isValid = true;
		GfxRenderState* renderState;
		GfxVertexBuffer* vertexBuffer;
		GfxIndexBuffer* indexBuffer;
		Topology topology;
		UINT indexCount;
		UINT indexOffset;
		UINT vertexCount;
		ObjectId materialId;
	};
}