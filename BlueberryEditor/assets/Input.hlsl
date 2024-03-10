cbuffer PerDrawData
{
	float4x4 modelMatrix;
}

cbuffer PerCameraData
{
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	float4x4 viewProjectionMatrix;
	float4x4 inverseViewMatrix;
	float4x4 inverseProjectionMatrix;
	float4x4 inverseViewProjectionMatrix;
	float4 cameraPositionWS;
	float4 cameraForwardDirectionWS;
};