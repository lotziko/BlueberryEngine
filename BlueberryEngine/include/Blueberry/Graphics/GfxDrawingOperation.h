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
		GfxDrawingOperation(GfxBuffer* vertexBuffer, GfxBuffer* indexBuffer, Material* material, VertexLayout* layout, uint32_t indexCount, uint32_t indexOffset, uint32_t vertexCount, Topology topology, uint8_t passIndex = 0, GfxBuffer* instanceBuffer = nullptr, uint32_t instanceOffset = 0, uint32_t instanceCount = 1, bool isCounterClockwise = false);
		GfxDrawingOperation(Mesh* mesh, Material* material, uint32_t indexCount, uint32_t indexOffset, uint32_t vertexCount, uint8_t passIndex = 0, GfxBuffer* instanceBuffer = nullptr, uint32_t instanceOffset = 0, uint32_t instanceCount = 1, bool isCounterClockwise = false);
		GfxDrawingOperation(Mesh* mesh, GfxBuffer* vertexBufferOverride, Material* material, uint32_t indexCount, uint32_t indexOffset, uint32_t vertexCount, uint8_t passIndex = 0, GfxBuffer* instanceBuffer = nullptr, uint32_t instanceOffset = 0, uint32_t instanceCount = 1, bool isCounterClockwise = false);
		GfxDrawingOperation(Mesh* mesh, Material* material, uint8_t passIndex = 0, GfxBuffer* instanceBuffer = nullptr, uint32_t instanceOffset = 0, uint32_t instanceCount = 1, bool isCounterClockwise = false);
		GfxDrawingOperation(Mesh* mesh, GfxBuffer* vertexBufferOverride, Material* material, uint8_t passIndex = 0, GfxBuffer* instanceBuffer = nullptr, uint32_t instanceOffset = 0, uint32_t instanceCount = 1, bool isCounterClockwise = false);

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
		bool isCounterClockwise;
	};
}