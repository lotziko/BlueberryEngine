#include "Blueberry\Scene\LightingData.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"

namespace Blueberry
{
	static GfxBuffer* s_ScaleOffsetBuffer = nullptr;

	OBJECT_DEFINITION(LightingData, Object)
	{
		DEFINE_FIELD(LightingData, m_Lightmap, BindingType::ObjectPtr, FieldOptions().SetObjectType(Texture2D::Type))
		DEFINE_FIELD(LightingData, m_ChartScaleOffset, BindingType::ByteData, {})
		DEFINE_FIELD(LightingData, m_ChartInstanceOffset, BindingType::ByteData, {})
	}

	Vector4* LightingData::GetChartScaleOffset()
	{
		return reinterpret_cast<Vector4*>(m_ChartScaleOffset.data());
	}

	void LightingData::SetChartScaleOffset(const List<Vector4>& scaleOffset)
	{
		m_ChartScaleOffset.resize(scaleOffset.size() * sizeof(Vector4));
		memcpy(m_ChartScaleOffset.data(), scaleOffset.data(), m_ChartScaleOffset.size());
	}

	void LightingData::SetInstanceOffset(const Dictionary<ObjectId, uint32_t>& instanceOffset)
	{
		m_ChartInstanceOffset.resize((sizeof(FileId) + sizeof(uint32_t)) * instanceOffset.size());
		uint8_t* ptr = m_ChartInstanceOffset.data();
		for (auto& pair : instanceOffset)
		{
			FileId fileId = ObjectDB::GetFileIdFromObjectId(pair.first);
			memcpy(ptr, &fileId, sizeof(FileId));
			ptr += sizeof(FileId);
			memcpy(ptr, &pair.second, sizeof(uint32_t));
			ptr += sizeof(uint32_t);
		}
	}

	Texture2D* LightingData::GetLightmap()
	{
		return m_Lightmap.Get();
	}

	void LightingData::SetLightmap(Texture2D* lightmap)
	{
		m_Lightmap = lightmap;
	}

	void LightingData::Apply(Scene* scene)
	{
		if (m_Lightmap.IsValid())
		{
			Texture2D* lightmap = m_Lightmap.Get();
			GfxDevice::SetGlobalTexture(TO_HASH("_LightmapTexture"), lightmap->Get());
		}
		else
		{
			return;
		}

		if (m_ChartScaleOffset.size() > 0)
		{
			if (s_ScaleOffsetBuffer != nullptr)
			{
				delete s_ScaleOffsetBuffer;
			}

			BufferProperties properties = {};
			properties.type = BufferType::Structured;
			properties.elementCount = m_ChartScaleOffset.size() / sizeof(Vector4);
			properties.elementSize = sizeof(Vector4);
			properties.data = m_ChartScaleOffset.data();
			properties.dataSize = m_ChartScaleOffset.size() * sizeof(Vector4);

			GfxDevice::CreateBuffer(properties, s_ScaleOffsetBuffer);
			GfxDevice::SetGlobalBuffer(TO_HASH("_PerLightmapInstanceData"), s_ScaleOffsetBuffer);
		}
		else
		{
			return;
		}

		if (m_ChartInstanceOffset.size() > 0)
		{
			Dictionary<FileId, uint32_t> offsets = {};
			uint8_t* ptr = m_ChartInstanceOffset.data();
			uint32_t count = (m_ChartInstanceOffset.size()) / (sizeof(FileId) + sizeof(uint32_t));
			for (uint32_t i = 0; i < count; ++i)
			{
				FileId fileId = *reinterpret_cast<FileId*>(ptr);
				ptr += sizeof(FileId);
				uint32_t offset = *reinterpret_cast<uint32_t*>(ptr);
				ptr += sizeof(uint32_t);
				offsets.insert({ fileId, offset });
			}
			// TODO find faster way, maybe use direct references to scene instead of FileId
			auto& obj = ObjectDB::GetObjectsFromGuid(Guid());
			List<Object*> objects;
			ObjectDB::GetObjects(MeshRenderer::Type, objects);
			for (Object* object : objects)
			{
				FileId fileId = ObjectDB::GetFileIdFromObjectId(object->GetObjectId());
				auto it = offsets.find(fileId);
				if (it != offsets.end())
				{
					MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(object);
					meshRenderer->SetLightmapChartOffset(it->second);
				}
			}
		}
		else
		{
			return;
		}

		Shader::SetKeyword(TO_HASH("LIGHTMAP"), true);
	}
}
