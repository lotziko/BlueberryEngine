#include "LightingData.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\ReflectionProbe.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"

#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\EditorSceneManager.h"

namespace Blueberry
{
	static GfxBuffer* s_ScaleOffsetBuffer = nullptr;
	static GfxTexture* s_ReflectionTexture = nullptr;

	#define REFLECTION_SIZE 128
	
	DATA_DEFINITION(MeshRendererData)
	{
		DEFINE_FIELD(MeshRendererData, m_ObjectId, BindingType::Raw, FieldOptions().SetSize(sizeof(GlobalObjectId)))
		DEFINE_FIELD(MeshRendererData, m_ChartInstanceOffset, BindingType::Int, {})
	}

	DATA_DEFINITION(ReflectionProbeData)
	{
		DEFINE_FIELD(ReflectionProbeData, m_ObjectId, BindingType::Raw, FieldOptions().SetSize(sizeof(GlobalObjectId)))
		DEFINE_FIELD(ReflectionProbeData, m_TextureCube, BindingType::ObjectPtr, FieldOptions().SetObjectType(TextureCube::Type))
	}

	OBJECT_DEFINITION(LightingData, Object)
	{
		DEFINE_PREFER_BINARY()
		DEFINE_FIELD(LightingData, m_Lightmap, BindingType::ObjectPtr, FieldOptions().SetObjectType(Texture2D::Type))
		DEFINE_FIELD(LightingData, m_ChartScaleOffset, BindingType::ByteData, {})
		DEFINE_FIELD(LightingData, m_ChartInstanceOffset, BindingType::ByteData, {})
		DEFINE_FIELD(LightingData, m_MeshRenderers, BindingType::DataList, FieldOptions().SetObjectType(MeshRendererData::Type))
		DEFINE_FIELD(LightingData, m_ReflectionProbes, BindingType::DataList, FieldOptions().SetObjectType(ReflectionProbeData::Type))
		DEFINE_FIELD(LightingData, m_ChartOffsetScale, BindingType::Vector4List, {})
	}

	const GlobalObjectId& MeshRendererData::GetObjectId()
	{
		return m_ObjectId;
	}

	void MeshRendererData::SetObjectId(const GlobalObjectId& objectId)
	{
		m_ObjectId = objectId;
	}

	const uint32_t& MeshRendererData::GetChartInstanceOffset()
	{
		return m_ChartInstanceOffset;
	}

	void MeshRendererData::SetChartInstanceOffset(const uint32_t& chartInstanceOffset)
	{
		m_ChartInstanceOffset = chartInstanceOffset;
	}

	const GlobalObjectId& ReflectionProbeData::GetObjectId()
	{
		return m_ObjectId;
	}

	void ReflectionProbeData::SetObjectId(const GlobalObjectId& objectId)
	{
		m_ObjectId = objectId;
	}

	TextureCube* ReflectionProbeData::GetTextureCube()
	{
		return m_TextureCube.Get();
	}

	void ReflectionProbeData::SetTextureCube(TextureCube* textureCube)
	{
		m_TextureCube = textureCube;
	}

	Vector4* LightingData::GetChartScaleOffset()
	{
		return reinterpret_cast<Vector4*>(m_ChartScaleOffset.data());
	}

	Texture2D* LightingData::GetLightmap()
	{
		return m_Lightmap.Get();
	}

	GlobalObjectId GetGlobalObjectId(Entity* entity)
	{
		GlobalObjectId id = {};
		if (PrefabManager::IsPartOfPrefabInstance(entity))
		{
			PrefabInstance* instance = PrefabManager::GetInstance(entity);
			id.prefabInstanceFileId = ObjectDB::GetFileIdFromObject(instance);
		}
		id.objectFileId = ObjectDB::GetFileIdFromObject(entity);
		return id;
	}

	void LightingData::SetLightmapData(Texture2D* lightmap, const List<Vector4>& scaleOffset, const Dictionary<ObjectId, uint32_t>& instanceOffset)
	{
		m_Lightmap = lightmap;
		m_ChartScaleOffset.resize(scaleOffset.size() * sizeof(Vector4));
		memcpy(m_ChartScaleOffset.data(), scaleOffset.data(), m_ChartScaleOffset.size());

		m_MeshRenderers.clear();
		for (auto& pair : instanceOffset)
		{
			MeshRendererData data = {};
			MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(ObjectDB::GetObject(pair.first));
			data.SetObjectId(GetGlobalObjectId(meshRenderer->GetEntity()));
			data.SetChartInstanceOffset(pair.second);
			m_MeshRenderers.push_back(data);
		}
	}

	uint32_t LightingData::GetReflectionProbeIndex(TextureCube* probeTexture)
	{
		for (size_t i = 0; i < m_ReflectionProbes.size(); ++i)
		{
			if (m_ReflectionProbes[i].GetTextureCube() == probeTexture)
			{
				return i;
			}
		}
		return UINT_MAX;
	}

	const size_t LightingData::GetReflectionProbeCount()
	{
		return m_ReflectionProbes.size();
	}

	void LightingData::SetSkyReflection(SkyRenderer* skyRenderer)
	{
		if (m_ReflectionProbes.size() == 0)
		{
			m_ReflectionProbes.resize(1);
		}

		ReflectionProbeData data = {};
		data.SetObjectId(GetGlobalObjectId(skyRenderer->GetEntity()));
		data.SetTextureCube(skyRenderer->GetReflectionTexture());
		m_ReflectionProbes[0] = data;
		ApplyReflections();
	}

	void LightingData::SetReflectionProbe(const uint32_t& index, ReflectionProbe* reflectionProbe)
	{
		if (m_ReflectionProbes.size() <= 1 + index)
		{
			m_ReflectionProbes.resize(1 + index + 1);
		}

		ReflectionProbeData data = {};
		data.SetObjectId(GetGlobalObjectId(reflectionProbe->GetEntity()));
		data.SetTextureCube(reflectionProbe->GetReflectionTexture());
		m_ReflectionProbes[1 + index] = data;
		ApplyReflections();
	}

	void LightingData::Apply()
	{
		ApplyLightmap();
		ApplyReflections();
	}

	void LightingData::ApplyLightmap()
	{
		if (m_ChartInstanceOffset.size() > 0 && m_MeshRenderers.size() == 0)
		{
			uint8_t* ptr = m_ChartInstanceOffset.data();
			uint32_t rendererCount = (m_ChartInstanceOffset.size()) / (sizeof(GlobalObjectId) + sizeof(uint32_t));

			m_MeshRenderers.resize(rendererCount);
			for (uint32_t i = 0; i < rendererCount; ++i)
			{
				GlobalObjectId objectId = *reinterpret_cast<GlobalObjectId*>(ptr);
				ptr += sizeof(GlobalObjectId);
				uint32_t offset = *reinterpret_cast<uint32_t*>(ptr);
				ptr += sizeof(uint32_t);

				MeshRendererData data = {};
				data.SetObjectId(objectId);
				data.SetChartInstanceOffset(offset);
				m_MeshRenderers[i] = data;
			}

			m_ChartOffsetScale.resize(m_ChartScaleOffset.size() / 4);
			memcpy(m_ChartOffsetScale.data(), m_ChartScaleOffset.data(), m_ChartScaleOffset.size());
		}

		if (m_Lightmap.IsValid() && m_Lightmap->GetState() == ObjectState::Default)
		{
			Texture2D* lightmap = m_Lightmap.Get();
			GfxDevice::SetGlobalTexture(TO_HASH("_LightmapTexture"), lightmap->Get());
		}
		else
		{
			Shader::SetKeyword(TO_HASH("LIGHTMAP"), false);
			return;
		}

		if (m_ChartOffsetScale.size() > 0)
		{
			if (s_ScaleOffsetBuffer != nullptr)
			{
				delete s_ScaleOffsetBuffer;
			}

			BufferProperties properties = {};
			properties.type = BufferType::Structured;
			properties.elementCount = static_cast<uint32_t>(m_ChartOffsetScale.size());
			properties.elementSize = sizeof(Vector4);
			properties.data = m_ChartOffsetScale.data();
			properties.dataSize = m_ChartOffsetScale.size() * sizeof(Vector4);

			GfxDevice::CreateBuffer(properties, s_ScaleOffsetBuffer);
			GfxDevice::SetGlobalBuffer(TO_HASH("_PerLightmapInstanceData"), s_ScaleOffsetBuffer);
		}
		else
		{
			Shader::SetKeyword(TO_HASH("LIGHTMAP"), false);
			return;
		}

		if (m_MeshRenderers.size() > 0)
		{
			Dictionary<FileId, Entity*> entities = {};
			List<Object*> sceneObjects;
			ObjectDB::GetObjects(Entity::Type, sceneObjects, SearchObjectType::WithoutGuid);

			for (Object* object : sceneObjects)
			{
				Entity* entity = static_cast<Entity*>(object);
				if (PrefabManager::IsPartOfPrefabInstance(entity))
				{
					continue;
				}
				FileId id = ObjectDB::GetFileIdFromObject(object);
				if (id != 0)
				{
					entities.insert_or_assign(id, entity);
				}
			}

			Dictionary<FileId, PrefabInstance*> prefabs = {};
			sceneObjects.clear();
			ObjectDB::GetObjects(PrefabInstance::Type, sceneObjects, SearchObjectType::WithoutGuid);

			for (Object* object : sceneObjects)
			{
				PrefabInstance* instance = static_cast<PrefabInstance*>(object);
				prefabs.insert_or_assign(ObjectDB::GetFileIdFromObject(object), instance);
			}

			uint32_t meshRendererCount = static_cast<uint32_t>(m_MeshRenderers.size());

			Guid guid = EditorSceneManager::GetGuid();
			for (uint32_t i = 0; i < meshRendererCount; ++i)
			{
				MeshRendererData data = m_MeshRenderers[i];
				GlobalObjectId id = data.GetObjectId();
				uint32_t offset = data.GetChartInstanceOffset();
				Entity* entity = nullptr;
				if (id.prefabInstanceFileId != 0)
				{
					auto it = prefabs.find(id.prefabInstanceFileId);
					if (it != prefabs.end())
					{
						PrefabInstance* instance = it->second;
						entity = instance->GetEntity(id.objectFileId);
					}
				}
				else
				{
					entity = entities.find(id.objectFileId)->second;
				}
				if (entity != nullptr)
				{
					entity->GetComponent<MeshRenderer>()->SetLightmapChartOffset(offset);
				}
			}
		}
		else
		{
			Shader::SetKeyword(TO_HASH("LIGHTMAP"), false);
			return;
		}

		Shader::SetKeyword(TO_HASH("LIGHTMAP"), true);
	}

	void LightingData::ApplyReflections()
	{
		uint32_t probeCount = static_cast<uint32_t>(m_ReflectionProbes.size());
		uint32_t textureSize = 0;
		List<uint8_t> textureData = {};

		if (probeCount > 0)
		{
			Dictionary<FileId, Entity*> entities = {};
			List<Object*> sceneObjects;
			ObjectDB::GetObjects(Entity::Type, sceneObjects, SearchObjectType::WithoutGuid);

			for (Object* object : sceneObjects)
			{
				Entity* entity = static_cast<Entity*>(object);
				if (PrefabManager::IsPartOfPrefabInstance(entity))
				{
					continue;
				}
				FileId id = ObjectDB::GetFileIdFromObject(object);
				if (id != 0)
				{
					entities.insert_or_assign(id, entity);
				}
			}

			Dictionary<FileId, PrefabInstance*> prefabs = {};
			sceneObjects.clear();
			ObjectDB::GetObjects(PrefabInstance::Type, sceneObjects, SearchObjectType::WithoutGuid);

			for (Object* object : sceneObjects)
			{
				PrefabInstance* instance = static_cast<PrefabInstance*>(object);
				prefabs.insert_or_assign(ObjectDB::GetFileIdFromObject(object), instance);
			}

			for (uint32_t i = 0; i < probeCount; ++i)
			{
				ReflectionProbeData data = m_ReflectionProbes[i];
				TextureCube* texture = data.GetTextureCube();
				if (texture != nullptr)
				{
					textureSize = texture->GetDataSize();
					textureData.resize(textureSize * probeCount);
					break;
				}
			}

			for (uint32_t i = 0; i < probeCount; ++i)
			{
				ReflectionProbeData data = m_ReflectionProbes[i];
				GlobalObjectId id = data.GetObjectId();
				Entity* entity = nullptr;
				if (id.prefabInstanceFileId != 0)
				{
					auto it = prefabs.find(id.prefabInstanceFileId);
					if (it != prefabs.end())
					{
						PrefabInstance* instance = it->second;
						entity = instance->GetEntity(id.objectFileId);
					}
				}
				else
				{
					entity = entities.find(id.objectFileId)->second;
				}
				if (entity != nullptr && i > 0)
				{
					entity->GetComponent<ReflectionProbe>()->SetAtlasIndex(i);
				}

				TextureCube* texture = data.GetTextureCube();
				if (texture != nullptr)
				{
					memcpy(textureData.data() + textureSize * i, texture->GetData(), textureSize);
				}
			}
		}

		TextureProperties textureProperties = {};

		textureProperties.width = REFLECTION_SIZE;
		textureProperties.height = REFLECTION_SIZE;
		textureProperties.depth = std::max(1u, probeCount);
		textureProperties.antiAliasing = 1;
		textureProperties.mipCount = 6;
		textureProperties.format = TextureFormat::BC6H_UFloat;
		textureProperties.dimension = TextureDimension::TextureCubeArray;
		textureProperties.wrapMode = WrapMode::Clamp;
		textureProperties.filterMode = FilterMode::Bilinear;
		textureProperties.data = textureData.data();
		textureProperties.dataSize = textureData.size();

		GfxDevice::CreateTexture(textureProperties, s_ReflectionTexture);

		GfxDevice::SetGlobalTexture(TO_HASH("_ReflectionTexture"), s_ReflectionTexture);
	}
}
