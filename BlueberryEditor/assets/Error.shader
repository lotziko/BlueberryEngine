Shader
{
	Options
	{
		BlendSrc One
		BlendDst Zero
		ZWrite On
		Cull None
	}

	HLSLBEGIN
	#include "Input.hlsl"

	struct Attributes
	{
		float3 positionOS : POSITION;
		float3 normalOS	: NORMAL;
	};

	struct Varyings
	{
		float4 positionCS : SV_POSITION;
		float3 normalWS : TEXCOORD0;
	};

	Varyings Vertex(Attributes input)
	{
		Varyings output;
		output.positionCS = mul(mul(float4(input.positionOS, 1.0f), _ModelMatrix), _ViewProjectionMatrix);
		output.normalWS = normalize(mul(float4(input.normalOS, 0.0f), _ModelMatrix).xyz);
		return output;
	}

	float4 Fragment(Varyings input) : SV_TARGET
	{
		float lambert = dot(input.normalWS, -_CameraForwardDirectionWS);
		return float4(1.0 * lambert, 0.0, 1.0 * lambert, 1.0);
	}
	HLSLEND
}