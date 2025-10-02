#include "LightingData.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"

#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\EditorSceneManager.h"

namespace Blueberry
{
	static GfxBuffer* s_ScaleOffsetBuffer = nullptr;

	struct GlobalObjectId
	{
		FileId prefabInstanceFileId;
		FileId objectFileId;
	};

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
		m_ChartInstanceOffset.resize((sizeof(GlobalObjectId) + sizeof(uint32_t)) * instanceOffset.size());
		uint8_t* ptr = m_ChartInstanceOffset.data();
		for (auto& pair : instanceOffset)
		{
			GlobalObjectId objectId = {};
			MeshRenderer* meshRenderer = static_cast<MeshRenderer*>(ObjectDB::GetObject(pair.first));
			Entity* entity = meshRenderer->GetEntity();
			if (PrefabManager::IsPartOfPrefabInstance(entity))
			{
				PrefabInstance* instance = PrefabManager::GetInstance(entity);
				objectId.prefabInstanceFileId = ObjectDB::GetFileIdFromObject(instance);
			}
			objectId.objectFileId = ObjectDB::GetFileIdFromObject(entity);
			memcpy(ptr, &objectId, sizeof(GlobalObjectId));
			ptr += sizeof(GlobalObjectId);
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
			Shader::SetKeyword(TO_HASH("LIGHTMAP"), false);
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
			properties.elementCount = static_cast<uint32_t>(m_ChartScaleOffset.size() / sizeof(Vector4));
			properties.elementSize = sizeof(Vector4);
			properties.data = m_ChartScaleOffset.data();
			properties.dataSize = m_ChartScaleOffset.size() * sizeof(Vector4);

			GfxDevice::CreateBuffer(properties, s_ScaleOffsetBuffer);
			GfxDevice::SetGlobalBuffer(TO_HASH("_PerLightmapInstanceData"), s_ScaleOffsetBuffer);
		}
		else
		{
			Shader::SetKeyword(TO_HASH("LIGHTMAP"), false);
			return;
		}

		if (m_ChartInstanceOffset.size() > 0)
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

			uint8_t* ptr = m_ChartInstanceOffset.data();
			uint32_t count = (m_ChartInstanceOffset.size()) / (sizeof(GlobalObjectId) + sizeof(uint32_t));

			Guid guid = EditorSceneManager::GetGuid();
			for (uint32_t i = 0; i < count; ++i)
			{
				GlobalObjectId objectId = *reinterpret_cast<GlobalObjectId*>(ptr);
				ptr += sizeof(GlobalObjectId);
				uint32_t offset = *reinterpret_cast<uint32_t*>(ptr);
				ptr += sizeof(uint32_t);

				Entity* entity = nullptr;
				if (objectId.prefabInstanceFileId != 0)
				{
					auto it = prefabs.find(objectId.prefabInstanceFileId);
					if (it != prefabs.end())
					{
						PrefabInstance* instance = it->second;
						entity = instance->GetEntity(objectId.objectFileId);
					}
				}
				else
				{
					entity = entities.find(objectId.objectFileId)->second;
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
}
