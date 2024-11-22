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
			GfxTexture* Get();

			ObjectId* textureId;
			uint32_t textureSlot;
			uint32_t samplerSlot;
		};

		GfxVertexShader* vertexShader;
		GfxGeometryShader* geometryShader;
		GfxFragmentShader* fragmentShader;

		TextureInfo fragmentTextures[16];
		uint32_t fragmentTextureCount;

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
		GfxDrawingOperation(GfxVertexBuffer* vertexBuffer, GfxIndexBuffer* indexBuffer, Material* material, const uint32_t& indexCount, const uint32_t& indexOffset, const uint32_t& vertexCount, const Topology& topology, const uint8_t& passIndex = 0, GfxVertexBuffer* instanceBuffer = nullptr, const uint32_t& instanceOffset = 0, const uint32_t& instanceCount = 1);
		GfxDrawingOperation(Mesh* mesh, Material* material, const uint32_t& indexCount, const uint32_t& indexOffset, const uint32_t& vertexCount, const uint8_t& passIndex = 0, GfxVertexBuffer* instanceBuffer = nullptr, const uint32_t& instanceOffset = 0, const uint32_t& instanceCount = 1);
		GfxDrawingOperation(Mesh* mesh, Material* material, const uint8_t& passIndex = 0, GfxVertexBuffer* instanceBuffer = nullptr, const uint32_t& instanceOffset = 0, const uint32_t& instanceCount = 1);

		static Material* GetValidMaterial(Material* material);

		bool IsValid() const;

		bool isValid = true;
		GfxRenderState* renderState;
		GfxVertexBuffer* vertexBuffer;
		GfxIndexBuffer* indexBuffer;
		GfxVertexBuffer* instanceBuffer;
		Topology topology;
		uint32_t indexCount;
		uint32_t indexOffset;
		uint32_t vertexCount;
		uint32_t instanceOffset;
		uint32_t instanceCount;
		ObjectId materialId;
		uint32_t materialCRC;
	};
}