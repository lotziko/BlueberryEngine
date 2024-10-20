Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex BlitVertex
		#pragma fragment BlitFragment

		#include "Input.hlsl"

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

		Texture2D _BlitTexture;
		SamplerState _BlitTexture_Sampler;

		Varyings BlitVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		float4 BlitFragment(Varyings input) : SV_TARGET
		{
			return _BlitTexture.Sample(_BlitTexture_Sampler, input.texcoord);
		}
		HLSLEND
	}
}