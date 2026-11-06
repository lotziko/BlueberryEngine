#include "LightingData.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\LightingSettings.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\ReflectionProbe.h"
#include "Blueberry\Scene\Components\ProbeVolume.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\TextureCubeArray.h"
#include "Blueberry\Graphics\Texture3D.h"

#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\EditorSceneManager.h"

namespace Blueberry
{
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

	DATA_DEFINITION(ProbeVolumeData)
	{
		DEFINE_FIELD(ProbeVolumeData, m_ObjectId, BindingType::Raw, FieldOptions().SetSize(sizeof(GlobalObjectId)))
		DEFINE_FIELD(ProbeVolumeData, m_Info, BindingType::ObjectPtr, FieldOptions().SetObjectType(&ProbeVolumeInfo::Type))
	}

	OBJECT_DEFINITION(LightingData, Object)
	{
		DEFINE_PREFER_BINARY()
		DEFINE_FIELD(LightingData, m_Lightmap, BindingType::ObjectPtr, FieldOptions().SetObjectType(&Texture2D::Type))
		DEFINE_FIELD(LightingData, m_MeshRenderers, BindingType::DataList, FieldOptions().SetObjectType(&MeshRendererData::Type))
		DEFINE_FIELD(LightingData, m_ReflectionProbes, BindingType::DataList, FieldOptions().SetObjectType(&ReflectionProbeData::Type))
		DEFINE_FIELD(LightingData, m_ProbeVolumes, BindingType::DataList, FieldOptions().SetObjectType(&ProbeVolumeData::Type))
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

	TextureCube* ReflectionProbeData::GetTexture()
	{
		return m_TextureCube.Get();
	}

	void ReflectionProbeData::SetTexture(TextureCube* texture)
	{
		m_TextureCube = texture;
	}

	const GlobalObjectId& ProbeVolumeData::GetObjectId()
	{
		return m_ObjectId;
	}

	void ProbeVolumeData::SetObjectId(const GlobalObjectId& objectId)
	{
		m_ObjectId = objectId;
	}

	ProbeVolumeInfo* ProbeVolumeData::GetInfo()
	{
		return m_Info.Get();
	}

	void ProbeVolumeData::SetInfo(ProbeVolumeInfo* info)
	{
		m_Info = info;
	}

	LightingData::LightingData()
	{
		m_LightingSettings = Object::Create<LightingSettings>();
	}

	LightingData::~LightingData()
	{
		if (m_LightingSettings.IsValid())
		{
			Object::Destroy(m_LightingSettings.Get());
		}
	}

	const List<Vector4>& LightingData::GetChartOffsetScale()
	{
		return m_ChartOffsetScale;
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
		m_LightingSettings->IncrementLightmapUpdateCount();
	}

	List<TextureCube*> LightingData::GetReflectionProbes()
	{
		List<TextureCube*> result;
		for (size_t i = 0; i < m_ReflectionProbes.size(); ++i)
		{
			result.push_back(m_ReflectionProbes[i].GetTexture());
		}
		return result;
	}

	size_t LightingData::GetReflectionProbeCount()
	{
		return m_ReflectionProbes.size();
	}

	void LightingData::SetSkyReflection(SkyRenderer* skyRenderer, TextureCube* texture)
	{
		if (m_ReflectionProbes.size() == 0)
		{
			m_ReflectionProbes.resize(1);
		}

		ReflectionProbeData data = {};
		data.SetObjectId(GetGlobalObjectId(skyRenderer->GetEntity()));
		data.SetTexture(texture);
		m_ReflectionProbes[0] = data;
		m_LightingSettings->IncrementReflectionProbesUpdateCount();
	}

	void LightingData::SetReflectionProbe(uint32_t index, ReflectionProbe* reflectionProbe, TextureCube* textureCube)
	{
		if (m_ReflectionProbes.size() <= 1 + index)
		{
			m_ReflectionProbes.resize(1 + index + 1);
		}

		ReflectionProbeData data = {};
		data.SetObjectId(GetGlobalObjectId(reflectionProbe->GetEntity()));
		data.SetTexture(textureCube);
		m_ReflectionProbes[1 + index] = data;
		m_LightingSettings->IncrementReflectionProbesUpdateCount();
	}

	size_t LightingData::GetProbeVolumeCount()
	{
		return m_ProbeVolumes.size();
	}

	void LightingData::SetProbeVolume(uint32_t index, ProbeVolume* probeVolume, ProbeVolumeInfo* info)
	{
		if (m_ProbeVolumes.size() <= index)
		{
			m_ProbeVolumes.resize(index + 1);
		}

		ProbeVolumeData data = {};
		data.SetObjectId(GetGlobalObjectId(probeVolume->GetEntity()));
		data.SetInfo(info);
		m_ProbeVolumes[index] = data;
		m_LightingSettings->IncrementProbeVolumesUpdateCount();
	}

	void LightingData::Apply()
	{
		Dictionary<FileId, Entity*> entities = {};
		Dictionary<FileId, PrefabInstance*> prefabs = {};

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

		for (Object* object : ObjectDB::GetObjects(PrefabInstance::Type, SearchObjectType::Any))
		{
			PrefabInstance* instance = static_cast<PrefabInstance*>(object);
			prefabs.insert_or_assign(ObjectDB::GetFileIdFromObject(object), instance);
		}

		auto FindEntity = [entities, prefabs](GlobalObjectId id) -> Entity*
		{
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
			return entity;
		};

		if (!m_MeshRenderers.empty())
		{
			uint32_t meshRendererCount = static_cast<uint32_t>(m_MeshRenderers.size());
			for (uint32_t i = 0; i < meshRendererCount; ++i)
			{
				MeshRendererData data = m_MeshRenderers[i];
				GlobalObjectId id = data.GetObjectId();
				uint32_t offset = data.GetChartInstanceOffset();
				Entity* entity = FindEntity(id);
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

		if (!m_ProbeVolumes.empty())
		{
			size_t probeCount = m_ProbeVolumes.size();
			for (size_t i = 0; i < probeCount; ++i)
			{
				ProbeVolumeData data = m_ProbeVolumes[i];
				GlobalObjectId id = data.GetObjectId();
				Entity* entity = FindEntity(id);
				if (entity != nullptr)
				{
					ProbeVolume* probeVolume = entity->GetComponent<ProbeVolume>();
					probeVolume->SetIndex(static_cast<uint32_t>(i));
				}
			}
		}

		List<ObjectPtr<ProbeVolumeInfo>> probeVolumes;
		if (!m_ProbeVolumes.empty())
		{
			size_t volumeCount = m_ProbeVolumes.size();
			for (size_t i = 0; i < volumeCount; ++i)
			{
				ProbeVolumeData data = m_ProbeVolumes[i];
				GlobalObjectId id = data.GetObjectId();
				Entity* entity = FindEntity(id);
				if (entity != nullptr)
				{
					ProbeVolume* probeVolume = entity->GetComponent<ProbeVolume>();
					probeVolume->SetIndex(static_cast<uint32_t>(i));
				}
				probeVolumes.push_back(data.GetInfo());
			}
		}

		List<ObjectPtr<TextureCube>> reflectionProbes;
		if (!m_ReflectionProbes.empty())
		{
			size_t probeCount = m_ReflectionProbes.size();
			for (size_t i = 0; i < probeCount; ++i)
			{
				ReflectionProbeData data = m_ReflectionProbes[i];
				if (i > 0)
				{
					GlobalObjectId id = data.GetObjectId();
					Entity* entity = FindEntity(id);
					if (entity != nullptr)
					{
						ReflectionProbe* reflectionProbe = entity->GetComponent<ReflectionProbe>();
						reflectionProbe->SetIndex(static_cast<uint32_t>(i));
					}
				}
				reflectionProbes.push_back(data.GetTexture());
			}
		}

		m_LightingSettings->SetLightmap(m_Lightmap.Get());
		m_LightingSettings->SetChartOffsetScale(m_ChartOffsetScale);
		m_LightingSettings->SetProbeVolumes(probeVolumes);
		m_LightingSettings->SetReflectionProbes(reflectionProbes);
	}

	LightingSettings* LightingData::GetLightingSettings()
	{
		return m_LightingSettings.Get();
	}
}
