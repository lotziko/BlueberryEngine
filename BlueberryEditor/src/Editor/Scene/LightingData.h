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

		const uint32_t& GetChartInstanceOffset();
		void SetChartInstanceOffset(const uint32_t& chartInstanceOffset);

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

		Vector4* GetChartScaleOffset();

		Texture2D* GetLightmap();
		void SetLightmapData(Texture2D* lightmap, const List<Vector4>& scaleOffset, const Dictionary<ObjectId, uint32_t>& instanceOffset);

		uint32_t GetReflectionProbeIndex(TextureCube* probeTexture);
		const size_t GetReflectionProbeCount();
		void SetSkyReflection(SkyRenderer* skyRenderer);
		void SetReflectionProbe(const uint32_t& index, ReflectionProbe* reflectionProbe);
		
		void Apply();
		void ApplyLightmap();
		void ApplyReflections();

	private:
		ObjectPtr<Texture2D> m_Lightmap;
		List<MeshRendererData> m_MeshRenderers;
		List<ReflectionProbeData> m_ReflectionProbes;
		List<Vector4> m_ChartOffsetScale;

		ByteData m_ChartScaleOffset;
		ByteData m_ChartInstanceOffset;
	};
}