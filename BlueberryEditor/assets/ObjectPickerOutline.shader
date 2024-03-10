struct Attributes
{
	float3 positionOS : POSITION;
	float2 texcoord : TEXCOORD0;
};

struct Varyings
{
	float4 positionCS : SV_POSITION;
	float2 texcoord : TEXCOORD0;
};

Varyings Vertex(Attributes input)
{
	Varyings output;
	output.positionCS = float4(input.positionOS, 1.0f);
	output.texcoord = input.texcoord;
	return output;
}

Texture2D _BaseMap : TEXTURE: register(t0);
SamplerState _BaseMap_Sampler : SAMPLER: register(s0);

float4 Fragment(Varyings input) : SV_TARGET
{
	float2 offset = float2(3.0 / 1920.0, 3.0 / 1080.0);

	float4 sample1 = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord);

	float4 sample2 = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord + float2(offset.x, 0));
	float4 sample3 = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord - float2(offset.x, 0));
	float4 sample4 = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord + float2(0, offset.y));
	float4 sample5 = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord - float2(0, offset.y));
	float4 sample6 = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord + float2(offset.x, offset.y));
	float4 sample7 = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord - float2(offset.x, offset.y));
	float4 sample8 = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord + float2(offset.x, -offset.y));
	float4 sample9 = _BaseMap.Sample(_BaseMap_Sampler, input.texcoord - float2(offset.x, -offset.y));

	if (sample1.r < 0.2)
	{
		return float4(1, 165.0 / 255.0, 0, 1) * 
			(
				sample2.r > 0.2 || 
				sample3.r > 0.2 || 
				sample4.r > 0.2 || 
				sample5.r > 0.2 ||
				sample6.r > 0.2 ||
				sample7.r > 0.2 ||
				sample8.r > 0.2 ||
				sample9.r > 0.2
			);
	}

	return float4(0, 0, 0, 0);
}
