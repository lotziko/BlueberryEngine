#pragma once

#include "Blueberry\Graphics\Structs.h"

namespace Blueberry
{
	class Mesh;
	class Material;
	class GfxBuffer;
	class VertexLayout;
	enum class Topology;

	struct GfxDrawingOperation
	{
		GfxDrawingOperation(GfxBuffer* vertexBuffer, GfxBuffer* indexBuffer, Material* material, VertexLayout* layout, const uint32_t& indexCount, const uint32_t& indexOffset, const uint32_t& vertexCount, const Topology& topology, const uint8_t& passIndex = 0, GfxBuffer* instanceBuffer = nullptr, const uint32_t& instanceOffset = 0, const uint32_t& instanceCount = 1);
		GfxDrawingOperation(Mesh* mesh, Material* material, const uint32_t& indexCount, const uint32_t& indexOffset, const uint32_t& vertexCount, const uint8_t& passIndex = 0, GfxBuffer* instanceBuffer = nullptr, const uint32_t& instanceOffset = 0, const uint32_t& instanceCount = 1);
		GfxDrawingOperation(Mesh* mesh, Material* material, const uint8_t& passIndex = 0, GfxBuffer* instanceBuffer = nullptr, const uint32_t& instanceOffset = 0, const uint32_t& instanceCount = 1);

		static Material* GetValidMaterial(Material* material);

		bool IsValid() const;

		bool isValid = true;
		GfxBuffer* vertexBuffer;
		GfxBuffer* indexBuffer;
		GfxBuffer* instanceBuffer;
		VertexLayout* layout;
		uint32_t indexCount;
		uint32_t indexOffset;
		uint32_t vertexCount;
		uint32_t instanceOffset;
		uint32_t instanceCount;
		Material* material;
		Topology topology;
		uint8_t passIndex;
	};
}