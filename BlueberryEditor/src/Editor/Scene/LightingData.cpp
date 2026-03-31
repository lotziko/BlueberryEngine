#include "LightingData.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\ReflectionProbe.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\TextureCubeArray.h"
#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\DefaultTextures.h"

#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\EditorSceneManager.h"

namespace Blueberry
{
	static GfxBuffer* s_ScaleOffsetBuffer = nullptr;
	static GfxTexture* s_ReflectionTexture = nullptr;

	static size_t s_LightmapTextureId = TO_HASH("_LightmapTexture");
	static size_t s_ProbeVolumeTextureId = TO_HASH("_ProbeVolumeTexture");
	static size_t s_ReflectionTextureId = TO_HASH("_ReflectionTexture");
	static size_t s_PerLightmapInstanceDataId = TO_HASH("_PerLightmapInstanceData");

	#define REFLECTION_SIZE 128
	
	DATA_DEFINITION(MeshRendererData)
	{
		DEFINE_FIELD(MeshRendererData, m_ObjectId, BindingType::Raw, FieldOptions().SetSize(sizeof(GlobalObjectId)))
		DEFINE_FIELD(MeshRendererData, m_ChartInstanceOffset, BindingType::Int, FieldOptions())
	}

	DATA_DEFINITION(ReflectionProbeData)
	{
		DEFINE_FIELD(ReflectionProbeData, m_ObjectId, BindingType::Raw, FieldOptions().SetSize(sizeof(GlobalObjectId)))
		DEFINE_FIELD(ReflectionProbeData, m_TextureCube, BindingType::ObjectPtr, FieldOptions().SetObjectType(&TextureCube::Type))
	}

	OBJECT_DEFINITION(LightingData, Object)
	{
		DEFINE_PREFER_BINARY()
		DEFINE_FIELD(LightingData, m_Lightmap, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Texture2D::Type))
		DEFINE_FIELD(LightingData, m_ProbeVolume, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Texture3D::Type))
		DEFINE_FIELD(LightingData, m_ChartScaleOffset, BindingType::ByteData, FieldOptions())
		DEFINE_FIELD(LightingData, m_ChartInstanceOffset, BindingType::ByteData, FieldOptions())
		DEFINE_FIELD(LightingData, m_MeshRenderers, BindingType::DataList, FieldOptions().SetObjectType(&MeshRendererData::Type))
		DEFINE_FIELD(LightingData, m_ReflectionProbes, BindingType::DataList, FieldOptions().SetObjectType(&ReflectionProbeData::Type))
		DEFINE_FIELD(LightingData, m_ChartOffsetScale, BindingType::Vector4List, FieldOptions())
	}

	const GlobalObjectId& MeshRendererData::GetObjectId()
	{
		return m_ObjectId;
	}

	void MeshRendererData::SetObjectId(const GlobalObjectId& objectId)
	{
		m_ObjectId = objectId;
	}

	uint32_t MeshRendererData::GetChartInstanceOffset()
	{
		return m_ChartInstanceOffset;
	}

	void MeshRendererData::SetChartInstanceOffset(uint32_t chartInstanceOffset)
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

	Vector4* LightingData::GetChartOffsetScale()
	{
		return m_ChartOffsetScale.data();
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
			id.objectFileId = ObjectDB::GetFileIdFromObject(PrefabManager::GetCorrespondingPrefabObject(entity));
		}
		else
		{
			id.objectFileId = ObjectDB::GetFileIdFromObject(entity);
		}
		return id;
	}

	void LightingData::SetLightmapData(Texture2D* lightmap, const List<Vector4>& scaleOffset, const Dictionary<ObjectId, uint32_t>& instanceOffset)
	{
		m_Lightmap = lightmap;
		m_ChartOffsetScale.resize(scaleOffset.size());
		memcpy(m_ChartOffsetScale.data(), scaleOffset.data(), scaleOffset.size() * sizeof(Vector4));

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

	void LightingData::SetProbeVolumeData(Texture3D* probeVolume)
	{
		m_ProbeVolume = probeVolume;
	}

	size_t LightingData::GetReflectionProbeCount()
	{
		return m_ReflectionProbes.size();
	}

	void LightingData::SetSkyReflection(SkyRenderer* skyRenderer, TextureCube* textureCube)
	{
		if (m_ReflectionProbes.size() == 0)
		{
			m_ReflectionProbes.resize(1);
		}

		ReflectionProbeData data = {};
		data.SetObjectId(GetGlobalObjectId(skyRenderer->GetEntity()));
		data.SetTextureCube(textureCube);
		m_ReflectionProbes[0] = data;
		ApplyReflections();
	}

	void LightingData::SetReflectionProbe(uint32_t index, ReflectionProbe* reflectionProbe, TextureCube* textureCube)
	{
		if (m_ReflectionProbes.size() <= 1 + index)
		{
			m_ReflectionProbes.resize(1 + index + 1);
		}

		ReflectionProbeData data = {};
		data.SetObjectId(GetGlobalObjectId(reflectionProbe->GetEntity()));
		data.SetTextureCube(textureCube);
		m_ReflectionProbes[1 + index] = data;
		ApplyReflections();
	}

	void LightingData::Apply()
	{
		ApplyLightmap();
		ApplyProbeVolume();
		ApplyReflections();
	}

	void LightingData::ApplyLightmap()
	{
		if (m_Lightmap.IsValid())
		{
			Texture2D* lightmap = m_Lightmap.Get();
			GfxDevice::SetGlobalTexture(s_LightmapTextureId, lightmap->Get());
		}
		else
		{
			GfxDevice::SetGlobalTexture(s_LightmapTextureId, DefaultTextures::GetWhite2D()->Get());
			return;
		}

		if (m_ChartOffsetScale.size() > 0)
		{
			if (s_ScaleOffsetBuffer != nullptr)
			{
				delete s_ScaleOffsetBuffer;
			}

			BufferProperties properties = {};
			properties.elementCount = static_cast<uint32_t>(m_ChartOffsetScale.size());
			properties.elementSize = sizeof(Vector4);
			properties.data = m_ChartOffsetScale.data();
			properties.dataSize = m_ChartOffsetScale.size() * sizeof(Vector4);
			properties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource;

			GfxDevice::CreateBuffer(properties, s_ScaleOffsetBuffer);
			GfxDevice::SetGlobalBuffer(s_PerLightmapInstanceDataId, s_ScaleOffsetBuffer);
		}
		else
		{
			return;
		}

		if (m_MeshRenderers.size() > 0)
		{
			Dictionary<FileId, Entity*> entities = {};
			
			for (Object* object : ObjectDB::GetObjects(Entity::Type, SearchObjectType::WithoutGuid))
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
					PrefabInstance* instance = static_cast<PrefabInstance*>(ObjectDB::GetObjectFromGuid(Guid(), id.prefabInstanceFileId));
					if (instance != nullptr)
					{
						entity = static_cast<Entity*>(instance->GetCorrespondingObject(id.objectFileId));
					}
				}
				else
				{
					auto it = entities.find(id.objectFileId);
					if (it != entities.end())
					{
						entity = it->second;
					}
				}
				if (entity != nullptr)
				{
					MeshRenderer* meshRenderer = entity->GetComponent<MeshRenderer>();
					if (meshRenderer != nullptr)
					{
						meshRenderer->SetLightmapChartOffset(offset);
					}
				}
			}
		}
	}

	void LightingData::ApplyProbeVolume()
	{
		if (m_ProbeVolume.IsValid())
		{
			GfxDevice::SetGlobalTexture(s_ProbeVolumeTextureId, m_ProbeVolume->Get());
		}
		else
		{
			GfxDevice::SetGlobalTexture(s_ProbeVolumeTextureId, DefaultTextures::GetWhite3D()->Get());
		}
	}

	void LightingData::ApplyReflections()
	{
		size_t probeCount = m_ReflectionProbes.size();
		size_t textureSize = 0;
		List<uint8_t> textureData = {};

		if (probeCount > 0)
		{
			Dictionary<FileId, Entity*> entities = {};
			
			for (Object* object : ObjectDB::GetObjects(Entity::Type, SearchObjectType::WithoutGuid))
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
			
			for (Object* object : ObjectDB::GetObjects(PrefabInstance::Type, SearchObjectType::WithoutGuid))
			{
				PrefabInstance* instance = static_cast<PrefabInstance*>(object);
				prefabs.insert_or_assign(ObjectDB::GetFileIdFromObject(object), instance);
			}

			for (size_t i = 0; i < probeCount; ++i)
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

			for (size_t i = 0; i < probeCount; ++i)
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
						entity = static_cast<Entity*>(instance->GetCorrespondingObject(id.objectFileId));
					}
				}
				else
				{
					auto it = entities.find(id.objectFileId);
					if (it != entities.end())
					{
						entity = it->second;
					}
				}

				TextureCube* texture = data.GetTextureCube();
				if (entity != nullptr && i > 0)
				{
					ReflectionProbe* reflectionProbe = entity->GetComponent<ReflectionProbe>();
					reflectionProbe->SetAtlasIndex(static_cast<uint32_t>(i));
				}

				if (texture != nullptr)
				{
					memcpy(textureData.data() + textureSize * i, texture->GetData(), textureSize);
				}
			}
		}

		if (probeCount > 0)
		{
			TextureProperties textureProperties = {};

			textureProperties.width = REFLECTION_SIZE;
			textureProperties.height = REFLECTION_SIZE;
			textureProperties.depth = static_cast<uint32_t>(probeCount);
			textureProperties.antiAliasing = 1;
			textureProperties.mipCount = 6;
			textureProperties.format = TextureFormat::BC6H_UFloat;
			textureProperties.dimension = TextureDimension::TextureCubeArray;
			textureProperties.wrapMode = WrapMode::Clamp;
			textureProperties.filterMode = FilterMode::Bilinear;
			textureProperties.data = textureData.data();
			textureProperties.dataSize = textureData.size();

			if (s_ReflectionTexture != nullptr)
			{
				delete s_ReflectionTexture;
				s_ReflectionTexture = nullptr;
			}
			GfxDevice::CreateTexture(textureProperties, s_ReflectionTexture);

			GfxDevice::SetGlobalTexture(s_ReflectionTextureId, s_ReflectionTexture);
		}
		else
		{
			GfxDevice::SetGlobalTexture(s_ReflectionTextureId, DefaultTextures::GetBlackCubeArray()->Get());
		}
	}

	void LightingData::Clear()
	{
		GfxDevice::SetGlobalTexture(s_LightmapTextureId, DefaultTextures::GetWhite2D()->Get());
		GfxDevice::SetGlobalTexture(s_ProbeVolumeTextureId, DefaultTextures::GetWhite3D()->Get());
		GfxDevice::SetGlobalTexture(s_ReflectionTextureId, DefaultTextures::GetBlackCubeArray()->Get());
	}
}
