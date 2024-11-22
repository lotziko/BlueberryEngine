#ifndef INPUT_INCLUDED
#define INPUT_INCLUDED

#define MAIN_LIGHT_CASCADES 3
#define MAX_REALTIME_LIGHTS 128

struct PerDrawData
{
	float4x4 modelMatrix;
};

StructuredBuffer<PerDrawData> _PerDrawData;

cbuffer PerDrawData
{
	float4x4 _ModelMatrix;
}

cbuffer PerDrawDataInstanced
{
	float4x4 _ModelMatrixInstanced[128];
}

cbuffer PerCameraData
{
	float4x4 _ViewMatrix;
	float4x4 _ProjectionMatrix;
	float4x4 _ViewProjectionMatrix;
	float4x4 _InverseViewMatrix;
	float4x4 _InverseProjectionMatrix;
	float4x4 _InverseViewProjectionMatrix;
	float4 _CameraPositionWS;
	float4 _CameraForwardDirectionWS;
	float4 _CameraNearFarClipPlane;
	float4 _CameraSizeInvSize;
};

cbuffer PerCameraLightData
{
	float4 _MainLightColor;
	float4 _MainLightDirection;
	float4x4 _MainWorldToShadow[MAIN_LIGHT_CASCADES + 1]; // extra empty cascade for far distances
	float4 _MainShadowBounds[MAIN_LIGHT_CASCADES + 1];
	float4 _MainShadowCascades[MAIN_LIGHT_CASCADES];

	float4 _LightsCount;
	float4 _LightParam[MAX_REALTIME_LIGHTS]; // x greater than 0 - has shadow, w - light.range * light.range
	float4 _LightPosition[MAX_REALTIME_LIGHTS];
	float4 _LightColor[MAX_REALTIME_LIGHTS];
	float4 _LightAttenuation[MAX_REALTIME_LIGHTS];
	float4 _LightDirection[MAX_REALTIME_LIGHTS];
	float4x4 _WorldToShadow[MAX_REALTIME_LIGHTS];
	float4 _ShadowBounds[MAX_REALTIME_LIGHTS]; // stores shadowmap slice offsets to counter atlas offsets in _WorldToShadow, otherwise it's just (0, 0, 1, 1)
	float4 _Shadow3x3PCFTermC0;
	float4 _Shadow3x3PCFTermC1;
	float4 _Shadow3x3PCFTermC2;
	float4 _Shadow3x3PCFTermC3;
};

Texture2D _ScreenOcclusionTexture;
SamplerState _ScreenOcclusionTexture_Sampler;

Texture2D _ShadowTexture;
SamplerComparisonState _ShadowTexture_Sampler;

#endif