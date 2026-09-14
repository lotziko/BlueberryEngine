#ifndef STRUCTS_INCLUDED
#define STRUCTS_INCLUDED

struct InputData
{
	float3 positionWS;
	float3 positionVS;
	float3 normalWS;
	float3 normalGS; //geometric roughness
	float3 viewDirectionWS;
	float2 normalizedScreenSpaceUV;
	float2 renderTargetUV;
	float3 bakedGI;
};

struct SurfaceData
{
	float3 albedo;
	float alpha;
	float metallic;
	float roughness;
	float3 normalTS;
	float3 emission;
	float occlusion;
};

struct GBufferData
{
	float4 color : SV_Target0;
	float4 normalWS : SV_Target1;
	float4 orm : SV_Target2;
	float4 bakedGI : SV_Target3;
};

#endif