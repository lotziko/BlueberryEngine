#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Structs.h"

namespace Blueberry
{
	class Texture2D;
	class Texture3D;
	class TextureCube;
	class Scene;
	class SkyRenderer;
	class ReflectionProbe;

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

		TextureCube* GetTextureCube();
		void SetTextureCube(TextureCube* textureCube);

	private:
		GlobalObjectId m_ObjectId;
		ObjectPtr<TextureCube> m_TextureCube;
	};

	class BB_API LightingData : public Object
	{
		OBJECT_DECLARATION(LightingData)

	public:
		LightingData() = default;
		virtual ~LightingData() = default;

		const List<Vector4>& GetChartOffsetScale();

		Texture2D* GetLightmap();
		void SetLightmapData(Texture2D* lightmap, const List<Vector4>& scaleOffset, const Dictionary<ObjectId, uint32_t>& instanceOffset);
		
		Texture3D* GetProbeVolume();
		void SetProbeVolumeData(Texture3D* probeVolume);

		List<TextureCube*> GetReflectionProbes();
		size_t GetReflectionProbeCount();
		void SetSkyReflection(SkyRenderer* skyRenderer, TextureCube* textureCube);
		void SetReflectionProbe(uint32_t index, ReflectionProbe* reflectionProbe, TextureCube* textureCube);
		
		void Apply();
		void ApplyLightmap();
		void ApplyProbeVolume();
		void ApplyReflections();

		static void Clear();

	private:
		ObjectPtr<Texture2D> m_Lightmap;
		ObjectPtr<Texture3D> m_ProbeVolume;
		List<MeshRendererData> m_MeshRenderers;
		List<ReflectionProbeData> m_ReflectionProbes;
		List<Vector4> m_ChartOffsetScale;
	};
}