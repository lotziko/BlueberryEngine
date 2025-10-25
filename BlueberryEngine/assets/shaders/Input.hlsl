#ifndef INPUT_INCLUDED
#define INPUT_INCLUDED

#define MAIN_LIGHT_CASCADES 3
#define MAX_REALTIME_LIGHTS 128
#define MAX_VIEW_COUNT 2

struct PerDrawData
{
	float4x4 modelMatrix;
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
	float4 _RenderTargetSizeInvSize;
	float4 _FogNearFarClipPlane;
};

struct PointLightData
{
	float3 positionWS;
	float hasShadow;
	float3 positionVS;
	float hasFog;
	float3 color;
	float squareRange;
	float4 attenuation;				// z,w unused
	float4x4 worldToShadow[6];
	float4 shadowBounds[6];			// stores shadowmap slice offsets to counter atlas offsets in _WorldToShadow, otherwise it's just (0, 0, 1, 1)
};

struct SpotLightData
{
	float3 positionWS;
	float hasShadow;
	float3 positionVS;
	float hasFog;
	float3 color;
	float hasCookie;
	float4 attenuation;
	float3 directionWS;
	float range;
	float3 directionVS;
	float coneOuterAngle;
	float4x4 worldToShadow;
	float4 shadowBounds;
	float4x4 worldToCookie;
};

StructuredBuffer<PointLightData> _PointLightsData;
StructuredBuffer<SpotLightData> _SpotLightsData;

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

	float4 _LightsCount;		// x - point lights, y - spot lights
	float4 _Shadow3x3PCFTermC0;
	float4 _Shadow3x3PCFTermC1;
	float4 _Shadow3x3PCFTermC2;
	float4 _Shadow3x3PCFTermC3;
};

TEXTURE2D_X(_ScreenOcclusionTexture);
SAMPLER(_ScreenOcclusionTexture_Sampler);

TEXTURE2D(_ShadowTexture);
SAMPLER_CMP(_ShadowTexture_Sampler);

TEXTURE3D(_CookieTexture);
SAMPLER(_CookieTexture_Sampler);

TEXTURE3D(_VolumetricFogTexture);
SAMPLER(_VolumetricFogTexture_Sampler);

TEXTURECUBE(_ReflectionTexture);
SAMPLER(_ReflectionTexture_Sampler);

TEXTURE2D(_LightmapTexture);
SAMPLER(_LightmapTexture_Sampler);

TEXTURE2D_UINT(_LightIndexTexture);

TEXTURE2D(_BlueNoiseLUT);
SAMPLER(_BlueNoiseLUT_Sampler);

TEXTURE2D(_BRDFIntegrationLUT);
SAMPLER(_BRDFIntegrationLUT_Sampler);

#endif