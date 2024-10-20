Shader
{
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex ResolveMSAAVertex
		#pragma fragment ResolveMSAAFragment

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

		struct Output
		{
			float4 color : SV_TARGET;
			float depth : SV_DEPTH;
		};

		Varyings ResolveMSAAVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		Texture2DMS<float4, 4> _ScreenNormalTexture;
		Texture2DMS<float, 4> _ScreenDepthStencilTexture;

		Output ResolveMSAAFragment(Varyings input)
		{
			Output output;
			for (int i = 0; i < 4; ++i)
			{
				uint2 uv = uint2(input.texcoord.x * _CameraSizeInvSize.x, input.texcoord.y * _CameraSizeInvSize.y);
				output.color += _ScreenNormalTexture.Load(uv, i);
				output.depth += _ScreenDepthStencilTexture.Load(uv, i);
			}
			output.color /= 4;
			output.depth /= 4;
			return output;
		}
		HLSLEND
	}
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex ResolveMSAAVertex
		#pragma fragment ResolveMSAAFragment

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

		struct Output
		{
			float4 color : SV_TARGET;
		};

		Varyings ResolveMSAAVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		Texture2DMS<float4, 4> _ScreenColorTexture;

		Output ResolveMSAAFragment(Varyings input)
		{
			Output output;
			for (int i = 0; i < 4; ++i)
			{
				uint2 uv = uint2(input.texcoord.x * _CameraSizeInvSize.x, input.texcoord.y * _CameraSizeInvSize.y);
				output.color += _ScreenColorTexture.Load(uv, i);
			}
			output.color /= 4;
			// Gamma correction
			output.color = float4(pow(output.color.rgb, 1.0 / 2.2), output.color.a);
			return output;
		}
		HLSLEND
	}
}