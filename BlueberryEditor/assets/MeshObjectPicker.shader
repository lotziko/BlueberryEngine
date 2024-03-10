#include "Input.hlsl"

cbuffer PerObjectData
{
	float objectId;
}

struct Attributes
{
	float3 positionOS : POSITION;
};

struct Varyings
{
	float4 positionCS : SV_POSITION;
};

Varyings Vertex(Attributes input)
{
	Varyings output;
	output.positionCS = mul(mul(float4(input.positionOS, 1.0f), modelMatrix), viewProjectionMatrix);
	return output;
}

float4 Fragment(Varyings input) : SV_TARGET
{
	return float4(objectId, 0, 0, 1);
}
