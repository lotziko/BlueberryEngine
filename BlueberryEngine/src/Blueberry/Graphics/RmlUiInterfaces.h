#pragma once

#include "Blueberry\Core\Base.h"

#include <RmlUi\Core.h>
#include <RmlUi\Core\RenderInterface.h>

namespace Blueberry
{
	class GfxBuffer;

	struct RmlUiVertex
	{
		Vector2 position;
		Color color;
		Vector2 texcoord;
	};

	struct RmlUiGeometryData
	{
		List<RmlUiVertex> vertices;
		List<int> indices;
		size_t vertexOffset = 0;
		size_t indexOffset = 0;
		bool isValid = true;
	};

	class RmlUiRenderData
	{
	public:
		~RmlUiRenderData();

	private:
		GfxBuffer* m_VertexBuffer = nullptr;
		GfxBuffer* m_IndexBuffer = nullptr;
		size_t m_VertexCount = 0;
		size_t m_IndexCount = 0;
		List<size_t> m_EmptyGeometryIndexes;
		List<RmlUiGeometryData> m_Geometry;
		bool m_IsDirty = true;

		friend class RmlUiRenderInterface;
	};

	class RmlUiRenderInterface : public Rml::RenderInterface
	{
	public:
		virtual Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) final;
		virtual void RenderGeometry(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, Rml::TextureHandle texture) final;
		virtual void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) final;
		virtual Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) final;
		virtual Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i source_dimensions) final;
		virtual void ReleaseTexture(Rml::TextureHandle texture) final;
		virtual void EnableScissorRegion(bool enable) final;
		virtual void SetScissorRegion(Rml::Rectanglei region) final;
	};

	class RmlUiSystemInterface : public Rml::SystemInterface
	{

	};
}