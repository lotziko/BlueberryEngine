Shader
{
	Properties
	{
		Texture2D _BaseMap = "white"
	}

	Options
	{
		BlendSrc One
		BlendDst Zero
		ZWrite Off
		Cull None
	}

	HLSLBEGIN
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
	};

	Varyings Vertex(Attributes input)
	{
		Varyings output;
		output.positionCS = mul(mul(float4(input.positionOS, 1.0f), _ModelMatrix), _ViewProjectionMatrix);
		output.color = input.color;
		return output;
	}

	float4 Fragment(Varyings input) : SV_TARGET
	{
		return input.color;
	}
	HLSLEND
}