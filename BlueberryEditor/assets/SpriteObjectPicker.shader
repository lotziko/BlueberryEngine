#include "Input.hlsl"

struct Attributes
{
	float3 positionOS : POSITION;
	float4 color : COLOR;
	float2 texcoord : TEXCOORD0;
};

struct Varyings
{
	float4 positionCS : SV_POSITION;
	float4 color : COLOR;
	float2 texcoord : TEXCOORD0;
};

Varyings Vertex(Attributes input)
{
	Varyings output;
	output.positionCS = mul(float4(input.positionOS, 1.0f), viewProjectionMatrix);
	output.color = input.color;
	output.texcoord = input.texcoord;
	return output;
}

Texture2D _BaseMap : TEXTURE: register(t0);
SamplerState _BaseMap_Sampler : SAMPLER: register(s0);

float4 Fragment(Varyings input) : SV_TARGET
{
	float4 color = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord);
	return input.color * ceil(color.a);
}
