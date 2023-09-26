struct Attributes
{
	float3 positionOS : POSITION;
	float4 color : COLOR;
	float2 texcoord : TEXCOORD0;
};

struct Varyings
{
	float4 positionCS : SV_POSITION;
	float2 texcoord : TEXCOORD0;
};

cbuffer PerDrawData : register(b0)
{
	float4x4 viewProjectionMatrix;
};

Varyings Vertex(Attributes input)
{
	Varyings output;
	output.positionCS = float4(input.positionOS, 1.0f);
	output.texcoord = input.texcoord;
	return output;
}

float4 Fragment(Varyings input) : SV_TARGET
{
	return float4(0, 0, 0, 0);
}
