struct PS_INPUT
{
	float4 inPosition : SV_POSITION;
    float4 inColor : COLOR;
	float2 inTexCoord : TEXCOORD0;
};

Texture2D _BaseMap : TEXTURE : register(t0);
SamplerState _BaseMap_Sampler : SAMPLER : register(s0);

float4 main(PS_INPUT input) : SV_TARGET
{
    float3 color = _BaseMap.Sample(_BaseMap_Sampler, input.inTexCoord);
    return float4(color, 1.0f) * input.inColor;
}