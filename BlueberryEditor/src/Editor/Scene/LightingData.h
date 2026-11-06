#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Structs.h"

namespace Blueberry
{
	class Texture2D;
	class Texture3D;
	class TextureCube;
	class TextureCubeArray;
	class Scene;
	class SkyRenderer;
	class ReflectionProbe;
	class ProbeVolume;
	class ProbeVolumeInfo;
	class LightingSettings;

	struct BB_API GlobalObjectId
	{
		FileId prefabInstanceFileId;
		FileId objectFileId;
	};

	class BB_API MeshRendererData : public Data
	{
		DATA_DECLARATION(MeshRendererData)

	public:
		const GlobalObjectId& GetObjectId();
		void SetObjectId(const GlobalObjectId& objectId);

		uint32_t GetChartInstanceOffset();
		void SetChartInstanceOffset(uint32_t chartInstanceOffset);

	private:
		GlobalObjectId m_ObjectId;
		uint32_t m_ChartInstanceOffset;
	};

	class BB_API ReflectionProbeData : public Data
	{
		DATA_DECLARATION(ReflectionProbeData)

	public:
		const GlobalObjectId& GetObjectId();
		void SetObjectId(const GlobalObjectId& objectId);

		TextureCube* GetTexture();
		void SetTexture(TextureCube* texture);

	private:
		GlobalObjectId m_ObjectId;
		ObjectPtr<TextureCube> m_TextureCube;
	};

	class BB_API ProbeVolumeData : public Data
	{
		DATA_DECLARATION(ProbeVolumeData)

	public:
		const GlobalObjectId& GetObjectId();
		void SetObjectId(const GlobalObjectId& objectId);

		ProbeVolumeInfo* GetInfo();
		void SetInfo(ProbeVolumeInfo* info);

	private:
		GlobalObjectId m_ObjectId;
		ObjectPtr<ProbeVolumeInfo> m_Info;
	};

	class BB_API LightingData : public Object
	{
		OBJECT_DECLARATION(LightingData)

	public:
		LightingData();
		virtual ~LightingData();

		const List<Vector4>& GetChartOffsetScale();

		Texture2D* GetLightmap();
		void SetLightmapData(Texture2D* lightmap, const List<Vector4>& scaleOffset, const Dictionary<ObjectId, uint32_t>& instanceOffset);

		List<TextureCube*> GetReflectionProbes();
		size_t GetReflectionProbeCount();
		void SetSkyReflection(SkyRenderer* skyRenderer, TextureCube* texture);
		void SetReflectionProbe(uint32_t index, ReflectionProbe* reflectionProbe, TextureCube* texture);

		size_t GetProbeVolumeCount();
		void SetProbeVolume(uint32_t index, ProbeVolume* probeVolume, ProbeVolumeInfo* info);

		void Apply();

		LightingSettings* GetLightingSettings();

	private:
		ObjectPtr<Texture2D> m_Lightmap;
		List<MeshRendererData> m_MeshRenderers;
		List<ProbeVolumeData> m_ProbeVolumes;
		List<ReflectionProbeData> m_ReflectionProbes;
		List<Vector4> m_ChartOffsetScale;

		ObjectPtr<LightingSettings> m_LightingSettings;
	};
}