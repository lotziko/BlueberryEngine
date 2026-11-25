#ifndef INPUT_INCLUDED
#define INPUT_INCLUDED

#define MAIN_LIGHT_CASCADES 3
#define MAX_REALTIME_LIGHTS 128
#define MAX_VIEW_COUNT 2

#define REFLECTION_LOD_COUNT 7
#define REFLECTION_ROUGHEST_MIP 1
#define REFLECTION_ROUGHNESS_MIP_SCALE 1.2

StructuredBuffer<float4x4> _SkinningData;

struct PerDrawData
{
	float4x4 modelMatrix;
	float4x4 normalMatrix;
	float4 lightmapChartOffset;
};

StructuredBuffer<PerDrawData> _PerDrawData;
StructuredBuffer<float4> _PerLightmapInstanceData;

cbuffer PerDrawData
{
	float4x4 _ModelMatrix;
}

cbuffer PerCameraData
{
	uint4 _ViewCount;
	float4x4 _ViewMatrix[MAX_VIEW_COUNT];
	float4x4 _ProjectionMatrix[MAX_VIEW_COUNT];
	float4x4 _ViewProjectionMatrix[MAX_VIEW_COUNT];
	float4x4 _InverseViewMatrix[MAX_VIEW_COUNT];
	float4x4 _InverseProjectionMatrix[MAX_VIEW_COUNT];
	float4x4 _InverseViewProjectionMatrix[MAX_VIEW_COUNT];
	float4 _CameraPositionWS;
	float4 _CameraForwardDirectionWS;
	float4 _CameraNearFarClipPlane;
	float4 _CameraSizeInvSize;
	float4 _CameraColor;
	float4 _RenderTargetSizeInvSize;
	float4 _FogNearFarClipPlane;
	float4 _DebugValue;
};

struct PointLightData
{
	float3 positionWS;
	float3 positionVS;
	float3 color;
	float squareRange;
	float4 attenuation;				// z,w unused
	uint flags;						// 1 - has shadow, 2 - has fog
	uint shadowDataOffset;
};

struct SpotLightData
{
	float3 positionWS;
	float3 positionVS;
	float3 color;
	float4 attenuation;
	float3 directionWS;
	float range;
	float3 directionVS;
	float coneOuterAngle;
	float4x4 worldToCookie;
	uint flags;						// 1 - has shadow, 2 - has fog, 4 - has cookie
	uint shadowDataOffset;
	float dummy;
};

struct ShadowData
{
	float4x4 worldToShadow;
	float4 shadowBounds;	// stores shadowmap slice offsets to counter atlas offsets in _WorldToShadow, otherwise it's just (0, 0, 1, 1)
};

struct ProbeVolumeData
{
	float3 positionMinWS;
	float3 positionMinVS;
	float3 positionMaxWS;
	float3 positionMaxVS;
	float3 invSizeWS;
	uint3 topLevelSize;
	uint probeAtlasOffset;
	float dummy;
};

struct ReflectionProbeData
{
	float3 positionWS;
	float squareRange;
	float volume;
	float fade;
	float3 positionMinWS;	// x is used as range for sphere
	float3 positionMinVS;	// is used as positionVS for sphere
	float3 positionMaxWS;
	float3 positionMaxVS;
	uint index;
	uint type;		// 0 - sphere, 1 - box
};

StructuredBuffer<PointLightData> _PointLightsData;
StructuredBuffer<SpotLightData> _SpotLightsData;
StructuredBuffer<ShadowData> _ShadowsData;
StructuredBuffer<ProbeVolumeData> _ProbeVolumesData;
StructuredBuffer<ReflectionProbeData> _ReflectionProbesData;

cbuffer PerCameraLightData
{
	float3 _MainLightColor;
	float _MainLightHasShadow;
	float3 _MainLightDirection;
	float _MainLightHasFog;
	float4x4 _MainWorldToShadow[MAIN_LIGHT_CASCADES + 1]; // extra empty cascade for far distances
	float4 _MainShadowBounds[MAIN_LIGHT_CASCADES + 1];
	float4 _MainShadowCascades[MAIN_LIGHT_CASCADES];
	float4 _AmbientLightColor;
	// maybe put here clusters size

	uint4 _LightsCount;		// x - point lights, y - spot lights, z - probe volumes, w - reflection probes
	float4 _Shadow3x3PCFTermC0;
	float4 _Shadow3x3PCFTermC1;
	float4 _Shadow3x3PCFTermC2;
	float4 _Shadow3x3PCFTermC3;
};

TEXTURE2D_X_UINT(_ScreenOcclusionTexture);
SAMPLER(_ScreenOcclusionTexture_Sampler);

TEXTURE2D_X(_ScreenReflectionTexture);
SAMPLER(_ScreenReflectionTexture_Sampler);

TEXTURE2D(_ShadowTexture);
SAMPLER_CMP(_ShadowTexture_Sampler);

TEXTURE3D(_CookieTexture);
SAMPLER(_CookieTexture_Sampler);

TEXTURE3D(_VolumetricFogTexture);
SAMPLER(_VolumetricFogTexture_Sampler);

TEXTURE2D(_LightmapTexture);
SAMPLER(_LightmapTexture_Sampler);

TEXTURE3D(_ProbeVolumeTexture0);
TEXTURE3D(_ProbeVolumeTexture1);
TEXTURE3D(_ProbeVolumeTexture2);
TEXTURE3D(_ProbeVolumeTexture3);
TEXTURE3D(_ProbeVolumeTexture4);
TEXTURE3D(_ProbeVolumeTexture5);
SAMPLER(_ProbeVolumeTexture0_Sampler);

TEXTURECUBE_ARRAY(_ReflectionTexture);
SAMPLER(_ReflectionTexture_Sampler);

TEXTURE2D_UINT(_LightIndexTexture);

TEXTURE2D(_BlueNoiseLUT);
SAMPLER(_BlueNoiseLUT_Sampler);

TEXTURE2D(_BRDFIntegrationLUT);
SAMPLER(_BRDFIntegrationLUT_Sampler);

#endif