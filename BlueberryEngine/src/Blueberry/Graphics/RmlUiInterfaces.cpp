#include "RmlUiInterfaces.h"

#include "Blueberry\Core\ObjectDB.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Tools\ByteConverter.h"
#include "Blueberry\Graphics\RmlUiRenderer.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\VertexLayout.h"
#include "Blueberry\Graphics\Buffers\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\DefaultTextures.h"
#include "Blueberry\Tools\StringHelper.h"

namespace Blueberry
{
	static size_t s_UiTextureId = TO_HASH("_UiTexture");

	struct RmlTextureData
	{
		GfxTexture* gfxTexture = nullptr;
		ObjectPtr<Texture2D> texture;
	};

	RmlUiRenderData::~RmlUiRenderData()
	{
		if (m_VertexBuffer != nullptr)
		{
			delete m_VertexBuffer;
		}
		if (m_IndexBuffer != nullptr)
		{
			delete m_IndexBuffer;
		}
	}

	Rml::CompiledGeometryHandle RmlUiRenderInterface::CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices)
	{
		RmlUiRenderData* renderData = RmlUiRenderer::s_CurrentData;
		size_t index;
		if (renderData->m_EmptyGeometryIndexes.size() > 0)
		{
			index = renderData->m_EmptyGeometryIndexes.back();
			renderData->m_EmptyGeometryIndexes.pop_back();
		}
		else
		{
			index = renderData->m_Geometry.size();
			renderData->m_Geometry.emplace_back();
		}

		RmlUiGeometryData& geometryData = renderData->m_Geometry[index];
		geometryData.isValid = true;
		geometryData.vertices.resize(vertices.size());
		for (size_t i = 0; i < vertices.size(); ++i)
		{
			const Rml::Vertex& vertex = vertices[i];
			geometryData.vertices[i] = { Vector2(vertex.position.x, vertex.position.y), Color(vertex.colour.red / 255.0f, vertex.colour.green / 255.0f, vertex.colour.blue / 255.0f, vertex.colour.alpha / 255.0f), Vector2(vertex.tex_coord.x, vertex.tex_coord.y) };
		}
		renderData->m_VertexCount += vertices.size();

		geometryData.indices.resize(indices.size());
		for (size_t i = 0; i < indices.size(); ++i)
		{
			geometryData.indices[i] = indices[i];
		}
		renderData->m_IndexCount += indices.size();
		renderData->m_IsDirty = true;

		return static_cast<Rml::CompiledGeometryHandle>(index);
	}

	void RmlUiRenderInterface::RenderGeometry(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, Rml::TextureHandle texture)
	{
		// TODO try to batch until texture is changed
		RmlUiRenderData* renderData = RmlUiRenderer::s_CurrentData;
		if (renderData->m_IsDirty)
		{
			if (renderData->m_VertexBuffer == nullptr)
			{
				BufferProperties vertexBufferProperties = {};
				vertexBufferProperties.elementCount = 2048;
				vertexBufferProperties.elementSize = sizeof(RmlUiVertex);
				vertexBufferProperties.usageFlags = BufferUsageFlags::VertexBuffer | BufferUsageFlags::CPUWritable;
				GfxDevice::CreateBuffer(vertexBufferProperties, renderData->m_VertexBuffer);

				BufferProperties indexBufferProperties = {};
				indexBufferProperties.elementCount = 2048;
				indexBufferProperties.elementSize = sizeof(int);
				indexBufferProperties.usageFlags = BufferUsageFlags::IndexBuffer | BufferUsageFlags::CPUWritable;
				GfxDevice::CreateBuffer(indexBufferProperties, renderData->m_IndexBuffer);
			}

			char* vertexPtr = static_cast<char*>(renderData->m_VertexBuffer->Map());
			if (vertexPtr != nullptr)
			{
				size_t vertexOffset = 0;
				for (size_t i = 0; i < renderData->m_Geometry.size(); ++i)
				{
					RmlUiGeometryData& data = renderData->m_Geometry[i];
					if (data.isValid)
					{
						data.vertexOffset = vertexOffset;
						size_t dataSize = data.vertices.size() * sizeof(RmlUiVertex);
						memcpy(vertexPtr, data.vertices.data(), dataSize);
						vertexPtr += dataSize;
						vertexOffset += data.vertices.size();
					}
				}
				renderData->m_VertexBuffer->Unmap();
			}

			int* indexPtr = static_cast<int*>(renderData->m_IndexBuffer->Map());
			if (indexPtr != nullptr)
			{
				size_t indexOffset = 0;
				for (size_t i = 0; i < renderData->m_Geometry.size(); ++i)
				{
					RmlUiGeometryData& geometryData = renderData->m_Geometry[i];
					if (geometryData.isValid)
					{
						geometryData.indexOffset = indexOffset;
						indexOffset += geometryData.indices.size();
						for (size_t i = 0; i < geometryData.indices.size(); ++i)
						{
							*indexPtr = static_cast<int>(geometryData.vertexOffset + geometryData.indices[i]);
							++indexPtr;
						}
					}
				}
				renderData->m_IndexBuffer->Unmap();
			}

			renderData->m_IsDirty = false;
		}

		size_t index = static_cast<size_t>(geometry);
		RmlUiGeometryData& geometryData = renderData->m_Geometry[index];
		PerDrawDataConstantBuffer::BindData(Matrix::CreateTranslation(translation.x, translation.y, 0.0f));
		uint8_t passIndex = 0;
		if (texture != 0)
		{
			RmlTextureData* textureData = reinterpret_cast<RmlTextureData*>(texture);
			GfxTexture* uiTexture;
			if (textureData->gfxTexture == nullptr)
			{
				if (textureData->texture.IsValid() && textureData->texture->IsDefaultState())
				{
					uiTexture = textureData->texture->Get();
				}
				else
				{
					uiTexture = DefaultTextures::GetWhite2D()->Get();
				}
			}
			else
			{
				uiTexture = textureData->gfxTexture;
			}
			GfxDevice::SetGlobalTexture(s_UiTextureId, uiTexture);
			passIndex = 1;
		}
		GfxDevice::Draw(GfxDrawingOperation(renderData->m_VertexBuffer, renderData->m_IndexBuffer, RmlUiRenderer::s_Material, RmlUiRenderer::s_VertexLayout, static_cast<uint32_t>(geometryData.indices.size()), static_cast<uint32_t>(geometryData.indexOffset), static_cast<uint32_t>(geometryData.vertices.size()), Topology::TriangleList, passIndex));
	}

	void RmlUiRenderInterface::ReleaseGeometry(Rml::CompiledGeometryHandle geometry)
	{
		RmlUiRenderData* renderData = RmlUiRenderer::s_CurrentData;
		size_t index = static_cast<size_t>(geometry);
		RmlUiGeometryData& data = renderData->m_Geometry[index];
		data.isValid = false;
		renderData->m_VertexCount -= data.vertices.size();
		renderData->m_IndexCount -= data.indices.size();
		renderData->m_EmptyGeometryIndexes.push_back(index);
		renderData->m_IsDirty = true;
	}

	// make UIElement wrapper which has SetBackgroundImage() which will set property to "id:6" and resolve it with ObjectId
	Rml::TextureHandle RmlUiRenderInterface::LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source)
	{
		RmlTextureData* textureData = new RmlTextureData();
		if (StringHelper::StartsWith(source.c_str(), "guid:"))
		{
			Guid guid;
			ByteConverter::HexStringToBytes(source.data() + 5, guid.data, source.size() - 5);
			Object* object = ObjectDB::GetObjectFromGuid(guid, 1);
			if (object != nullptr && object->IsClassType(Texture2D::Type))
			{
				textureData->texture = static_cast<Texture2D*>(object);
			}
		}
		return reinterpret_cast<Rml::TextureHandle>(textureData);
	}

	Rml::TextureHandle RmlUiRenderInterface::GenerateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i source_dimensions)
	{
		RmlTextureData* textureData = new RmlTextureData();
		TextureProperties textureProperties = {};
		textureProperties.width = source_dimensions.x;
		textureProperties.height = source_dimensions.y;
		textureProperties.depth = 1;
		textureProperties.antiAliasing = 1;
		textureProperties.mipCount = 1;
		textureProperties.format = TextureFormat::R8G8B8A8_UNorm;
		textureProperties.dimension = TextureDimension::Texture2D;
		textureProperties.wrapMode = WrapMode::Clamp;
		textureProperties.filterMode = FilterMode::Bilinear;
		textureProperties.usageFlags = TextureUsageFlags::None;
		textureProperties.data = source.data();
		textureProperties.dataSize = source.size();
		GfxDevice::CreateTexture(textureProperties, textureData->gfxTexture);
		return reinterpret_cast<Rml::TextureHandle>(textureData);
	}

	void RmlUiRenderInterface::ReleaseTexture(Rml::TextureHandle texture)
	{
		RmlTextureData* textureData = reinterpret_cast<RmlTextureData*>(texture);
		if (textureData->gfxTexture != nullptr)
		{
			delete textureData->gfxTexture;
		}
		delete textureData;
	}

	void RmlUiRenderInterface::EnableScissorRegion(bool enable)
	{
	}

	void RmlUiRenderInterface::SetScissorRegion(Rml::Rectanglei region)
	{
	}
}