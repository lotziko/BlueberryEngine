#ifndef INPUT_INCLUDED
#define INPUT_INCLUDED

#define MAX_REALTIME_LIGHTS 128

cbuffer PerDrawData
{
	float4x4 _ModelMatrix;
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
};

cbuffer PerCameraLightData
{
	float4 _LightsCount;
	float4 _LightPosition[MAX_REALTIME_LIGHTS];
	float4 _LightColor[MAX_REALTIME_LIGHTS];
	float4 _LightAttenuation[MAX_REALTIME_LIGHTS];
	float4 _LightDirection[MAX_REALTIME_LIGHTS];
};

#endif