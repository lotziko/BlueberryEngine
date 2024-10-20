Shader
{
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex LinearToSRGBVertex
		#pragma fragment LinearToSRGBFragment

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

		Varyings LinearToSRGBVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		Texture2D _ScreenTexture;
		SamplerState _ScreenTexture_Sampler;

		float4 LinearToSRGBFragment(Varyings input) : SV_TARGET
		{
			float4 screenColor = _ScreenTexture.Sample(_ScreenTexture_Sampler, input.texcoord);
			return float4(pow(screenColor.rgb, 1.0 / 2.2), screenColor.a);
		}
		HLSLEND
	}
}