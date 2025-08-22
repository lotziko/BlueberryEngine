Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex PostProcessingVertex
		#pragma fragment PostProcessingFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
			VERTEX_INPUT_INSTANCE_ID
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
			VERTEX_OUTPUT_VIEW_INDEX
		};

		cbuffer _PostProcessingData
		{
			float4 _ExposureTime;
		};

		Varyings PostProcessingVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		float3 TonemapFilmic(in float3 color, float shoulderScale = 0.15, float linearScale = 0.5, float linearAngle = 0.1, float toeScale = 0.2, float toeNumerator = 0.02, float toeDenominator = 0.3, float whitePointScale = 1.2867)
		{
			color = (color * (shoulderScale * color + linearAngle * linearScale) + toeScale * toeNumerator) / (color * (shoulderScale * color + linearScale) + toeScale * toeDenominator) - (toeNumerator / toeDenominator);
			return color * whitePointScale;
		}

		TEXTURE2D(_ScreenColorTexture);	SAMPLER(_ScreenColorTexture_Sampler);

		float3 Uncharted2Tonemap(float3 x)
		{
			float A = 0.15;
			float B = 0.50;
			float C = 0.10;
			float D = 0.20;
			float E = 0.02;
			float F = 0.30;
			float W = 11.2;

			float3 curr = ((x * (A * x + C * B) + D * E) /
				(x * (A * x + B) + D * F)) - E / F;

			float whiteScale = ((W * (A * W + C * B) + D * E) /
				(W * (A * W + B) + D * F)) - E / F;

			return curr / whiteScale;
		}

		float3 LinearToSRGB(float3 color)
		{
			return color <= 0.0031308f	? color * 12.92f : pow(color, 1.0f / 2.4f) * 1.055f - 0.055f;
		}

		float4 PostProcessingFragment(Varyings input) : SV_TARGET
		{
			// TODO bloom
			float4 color = SAMPLE_TEXTURE2D_X(_ScreenColorTexture, _ScreenColorTexture_Sampler, input.texcoord);

			float3 mixedColor = color.rgb;
			float3 tonemappedColor = Uncharted2Tonemap(mixedColor * _ExposureTime.x);

			// Gamma correction
			tonemappedColor = LinearToSRGB(tonemappedColor);

			float3 blueNoise = (SAMPLE_TEXTURE2D(_BlueNoiseLUT, _BlueNoiseLUT_Sampler, input.texcoord * RENDER_TARGET_SIZE_INV_SIZE.xy / 256.0 + _ExposureTime.yy).rgb - 0.5) / 255;
			tonemappedColor += blueNoise;

			return float4(tonemappedColor, color.a);
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex PostProcessingVertex
		#pragma fragment PostProcessingFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
			VERTEX_INPUT_INSTANCE_ID
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
			VERTEX_OUTPUT_VIEW_INDEX
		};

		Varyings PostProcessingVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		TEXTURE2D(_ScreenColorTexture);	SAMPLER(_ScreenColorTexture_Sampler);

		float4 PostProcessingFragment(Varyings input) : SV_TARGET
		{
			float4 color = SAMPLE_TEXTURE2D_X(_ScreenColorTexture, _ScreenColorTexture_Sampler, input.texcoord);
			// Gamma correction
			color.rgb = pow(color.rgb, 1.0 / 2.2);
			return color;
		}
		HLSLEND
	}
}