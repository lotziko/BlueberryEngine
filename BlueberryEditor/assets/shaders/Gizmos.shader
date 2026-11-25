Shader
{
	HLSLINCLUDE
	cbuffer GizmoData
	{
		float4 _Color;
		float _Thickness;
		float3 _Dummy;
	};
	HLSLEND

	Pass	// Line/triangle less than depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex ColorVertex
		#pragma fragment ColorFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float4 color : COLOR;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float4 color : COLOR;
		};

		Varyings ColorVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(input.positionOS, 1.0f)));
			output.color = input.color;
			return output;
		}

		float4 ColorFragment(Varyings input) : SV_TARGET
		{
			return input.color;
		}
		HLSLEND
	}
	Pass	// Line/triangle greater than depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		ZTest Greater
		Cull None

		HLSLBEGIN
		#pragma vertex ColorVertex
		#pragma fragment ColorFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float4 color : COLOR;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float4 color : COLOR;
		};

		Varyings ColorVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(input.positionOS, 1.0f)));
			output.color = input.color;
			return output;
		}

		float4 ColorFragment(Varyings input) : SV_TARGET
		{
			input.color.rgb *= 0.25;
			return input.color;
		}
		HLSLEND
	}
	Pass	// Line/triangle ignoring depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		ZTest Always
		Cull None

		HLSLBEGIN
		#pragma vertex ColorVertex
		#pragma fragment ColorFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float4 color : COLOR;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float4 color : COLOR;
		};

		Varyings ColorVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(input.positionOS, 1.0f)));
			output.color = input.color;
			return output;
		}

		float4 ColorFragment(Varyings input) : SV_TARGET
		{
			return input.color;
		}
		HLSLEND
	}
	Pass	// Arc less than depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex ArcLineVertex
		#pragma geometry ArcLineGeometry
		#pragma fragment ArcLineFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: NORMAL;
			float4 fromPositionOSAngle	: TANGENT;
			float4 color				: COLOR;
		};

		struct GeometryVaryings
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: TEXCOORD0;
			float4 tangentOSAngle		: TEXCOORD1;
			float3 bitangentOS			: TEXCOORD2;
			float4 color				: TEXCOORD3;
		};

		struct FragmentVaryings
		{
			float4 positionCS	: SV_POSITION;
			float4 color		: TEXCOORD0;
		};

		GeometryVaryings ArcLineVertex(Attributes input)
		{
			GeometryVaryings output;
			output.centerPositionOS = input.centerPositionOS;
			output.normalOSRadius = input.normalOSRadius;
			output.tangentOSAngle = float4(normalize(input.fromPositionOSAngle.xyz), input.fromPositionOSAngle.w);
			output.bitangentOS = cross(output.normalOSRadius.xyz, output.tangentOSAngle.xyz);
			output.color = input.color;
			return output;
		}

		[maxvertexcount(128)]
		void ArcLineGeometry(point GeometryVaryings input[1], inout LineStream<FragmentVaryings> lineStream)
		{
			float3 center = input[0].centerPositionOS;
			float3 normalOS = input[0].normalOSRadius.xyz;
			float3 tangentOS = input[0].tangentOSAngle.xyz;
			float3 bitangentOS = input[0].bitangentOS;
			float radius = input[0].normalOSRadius.w;
			float angle = input[0].tangentOSAngle.w;
			float4 color = input[0].color;

			float segments = 36.0;
			for (int i = 0; i <= 36; ++i)
			{
				float u = cos(radians(i / segments * angle)) * radius;
				float v = sin(radians(i / segments * angle)) * radius;

				FragmentVaryings p1;
				p1.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(center + tangentOS * u + bitangentOS * v, 1.0)));
				p1.color = color;
				lineStream.Append(p1);
			}
			lineStream.RestartStrip();
		}

		float4 ArcLineFragment(FragmentVaryings input) : SV_TARGET
		{
			return input.color;
		}
		HLSLEND
	}
	Pass	// Arc greater than depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		ZTest Greater
		Cull None

		HLSLBEGIN
		#pragma vertex ArcLineVertex
		#pragma geometry ArcLineGeometry
		#pragma fragment ArcLineFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: NORMAL;
			float4 fromPositionOSAngle	: TANGENT;
			float4 color				: COLOR;
		};

		struct GeometryVaryings
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: TEXCOORD0;
			float4 tangentOSAngle		: TEXCOORD1;
			float3 bitangentOS			: TEXCOORD2;
			float4 color				: TEXCOORD3;
		};

		struct FragmentVaryings
		{
			float4 positionCS	: SV_POSITION;
			float4 color		: TEXCOORD0;
		};

		GeometryVaryings ArcLineVertex(Attributes input)
		{
			GeometryVaryings output;
			output.centerPositionOS = input.centerPositionOS;
			output.normalOSRadius = input.normalOSRadius;
			output.tangentOSAngle = float4(normalize(input.fromPositionOSAngle.xyz), input.fromPositionOSAngle.w);
			output.bitangentOS = cross(output.normalOSRadius.xyz, output.tangentOSAngle.xyz);
			output.color = input.color;
			return output;
		}

		[maxvertexcount(128)]
		void ArcLineGeometry(point GeometryVaryings input[1], inout LineStream<FragmentVaryings> lineStream)
		{
			float3 center = input[0].centerPositionOS;
			float3 normalOS = input[0].normalOSRadius.xyz;
			float3 tangentOS = input[0].tangentOSAngle.xyz;
			float3 bitangentOS = input[0].bitangentOS;
			float radius = input[0].normalOSRadius.w;
			float angle = input[0].tangentOSAngle.w;
			float4 color = input[0].color;

			float segments = 36.0;
			for (int i = 0; i <= 36; ++i)
			{
				float u = cos(radians(i / segments * angle)) * radius;
				float v = sin(radians(i / segments * angle)) * radius;

				FragmentVaryings p1;
				p1.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(center + tangentOS * u + bitangentOS * v, 1.0)));
				p1.color = color;
				lineStream.Append(p1);
			}
			lineStream.RestartStrip();
		}

		float4 ArcLineFragment(FragmentVaryings input) : SV_TARGET
		{
			input.color.rgb *= 0.25;
			return input.color;
		}
		HLSLEND
	}
	Pass	// Arc ignoring depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		ZTest Always
		Cull None

		HLSLBEGIN
		#pragma vertex ArcLineVertex
		#pragma geometry ArcLineGeometry
		#pragma fragment ArcLineFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: NORMAL;
			float4 fromPositionOSAngle	: TANGENT;
			float4 color				: COLOR;
		};

		struct GeometryVaryings
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: TEXCOORD0;
			float4 tangentOSAngle		: TEXCOORD1;
			float3 bitangentOS			: TEXCOORD2;
			float4 color				: TEXCOORD3;
		};

		struct FragmentVaryings
		{
			float4 positionCS	: SV_POSITION;
			float4 color		: TEXCOORD0;
		};

		GeometryVaryings ArcLineVertex(Attributes input)
		{
			GeometryVaryings output;
			output.centerPositionOS = input.centerPositionOS;
			output.normalOSRadius = input.normalOSRadius;
			output.tangentOSAngle = float4(normalize(input.fromPositionOSAngle.xyz), input.fromPositionOSAngle.w);
			output.bitangentOS = cross(output.normalOSRadius.xyz, output.tangentOSAngle.xyz);
			output.color = input.color;
			return output;
		}

		[maxvertexcount(128)]
		void ArcLineGeometry(point GeometryVaryings input[1], inout LineStream<FragmentVaryings> lineStream)
		{
			float3 center = input[0].centerPositionOS;
			float3 normalOS = input[0].normalOSRadius.xyz;
			float3 tangentOS = input[0].tangentOSAngle.xyz;
			float3 bitangentOS = input[0].bitangentOS;
			float radius = input[0].normalOSRadius.w;
			float angle = input[0].tangentOSAngle.w;
			float4 color = input[0].color;

			float segments = 36.0;
			for (int i = 0; i <= 36; ++i)
			{
				float u = cos(radians(i / segments * angle)) * radius;
				float v = sin(radians(i / segments * angle)) * radius;

				FragmentVaryings p1;
				p1.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(center + tangentOS * u + bitangentOS * v, 1.0)));
				p1.color = color;
				lineStream.Append(p1);
			}
			lineStream.RestartStrip();
		}

		float4 ArcLineFragment(FragmentVaryings input) : SV_TARGET
		{
			return input.color;
		}
		HLSLEND
	}
	Pass	// Mesh less than depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex ColorVertex
		#pragma fragment ColorFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
		};

		Varyings ColorVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(input.positionOS, 1.0f)));
			return output;
		}

		float4 ColorFragment(Varyings input) : SV_TARGET
		{
			return float4(1, 1, 1, 1) * _Color;
		}
		HLSLEND
	}
	Pass	// Mesh greater than depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		ZTest Greater
		Cull None

		HLSLBEGIN
		#pragma vertex ColorVertex
		#pragma fragment ColorFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
		};

		Varyings ColorVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(input.positionOS, 1.0f)));
			return output;
		}

		float4 ColorFragment(Varyings input) : SV_TARGET
		{
			return float4(0.25, 0.25, 0.25, 1) * _Color;
		}
		HLSLEND
	}
	Pass	// Mesh ignoring depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		ZTest Always
		Cull None

		HLSLBEGIN
		#pragma vertex ColorVertex
		#pragma fragment ColorFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
		};

		Varyings ColorVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(input.positionOS, 1.0f)));
			return output;
		}

		float4 ColorFragment(Varyings input) : SV_TARGET
		{
			return float4(1, 1, 1, 1) * _Color;
		}
		HLSLEND
	}
	Pass	// Thick line ignoring depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		ZTest Always
		Cull None

		HLSLBEGIN
		#pragma vertex ColorVertex
		#pragma geometry ColorGeometry
		#pragma fragment ColorFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float4 color : COLOR;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float4 color : COLOR;
		};

		Varyings ColorVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(input.positionOS, 1.0f)));
			output.color = input.color;
			return output;
		}

		[maxvertexcount(4)]
		void ColorGeometry(line Varyings input[2], inout TriangleStream<Varyings> stream)
		{
			float4 a = input[0].positionCS;
			float4 b = input[1].positionCS;

			float2 viewportSize = CAMERA_SIZE_INV_SIZE.xy;
			float2 ndcA = a.xy / a.w;
			float2 ndcB = b.xy / b.w;

			float2 pixelDirection = (ndcB - ndcA) * viewportSize;
			float lengthSquared = dot(pixelDirection, pixelDirection);
			pixelDirection *= rsqrt(lengthSquared);
			float2 perpendicular = float2(-pixelDirection.y, pixelDirection.x);
			float2 offsetA = perpendicular * _Thickness * CAMERA_SIZE_INV_SIZE.zw * a.w;
			float2 offsetB = perpendicular * _Thickness * CAMERA_SIZE_INV_SIZE.zw * b.w;

			Varyings v;

			v.color = input[0].color;
			v.positionCS = a;
			v.positionCS.xy -= offsetA;
			stream.Append(v);

			v.positionCS = a;
			v.positionCS.xy += offsetA;
			stream.Append(v);

			v.color = input[1].color;
			v.positionCS = b;
			v.positionCS.xy -= offsetB;
			stream.Append(v);

			v.positionCS = b;
			v.positionCS.xy += offsetB;
			stream.Append(v);

			stream.RestartStrip();
		}

		float4 ColorFragment(Varyings input) : SV_TARGET
		{
			return input.color;
		}
		HLSLEND
	}
	Pass	// Thick arc ignoring depth
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		ZTest Always
		Cull None

		HLSLBEGIN
		#pragma vertex ArcLineVertex
		#pragma geometry ArcLineGeometry
		#pragma fragment ArcLineFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: NORMAL;
			float4 fromPositionOSAngle	: TANGENT;
			float4 color				: COLOR;
		};

		struct GeometryVaryings
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: TEXCOORD0;
			float4 tangentOSAngle		: TEXCOORD1;
			float3 bitangentOS			: TEXCOORD2;
			float4 color				: TEXCOORD3;
		};

		struct FragmentVaryings
		{
			float4 positionCS	: SV_POSITION;
			float4 color		: TEXCOORD0;
		};

		GeometryVaryings ArcLineVertex(Attributes input)
		{
			GeometryVaryings output;
			output.centerPositionOS = input.centerPositionOS;
			output.normalOSRadius = input.normalOSRadius;
			output.tangentOSAngle = float4(normalize(input.fromPositionOSAngle.xyz), input.fromPositionOSAngle.w);
			output.bitangentOS = cross(output.normalOSRadius.xyz, output.tangentOSAngle.xyz);
			output.color = input.color;
			return output;
		}

		[maxvertexcount(128)]
		void ArcLineGeometry(point GeometryVaryings input[1], inout TriangleStream<FragmentVaryings> triangleStream)
		{
			float3 center = input[0].centerPositionOS;
			float3 normalOS = input[0].normalOSRadius.xyz;
			float3 tangentOS = input[0].tangentOSAngle.xyz;
			float3 bitangentOS = input[0].bitangentOS;
			float radius = input[0].normalOSRadius.w;
			float angle = input[0].tangentOSAngle.w;
			float4 color = input[0].color;

			float segments = 36.0;
			float stepRadians = radians(angle / segments);
			float2 viewportSize = CAMERA_SIZE_INV_SIZE.xy;
			float2 inverseViewportSize = CAMERA_SIZE_INV_SIZE.zw;

			for (int i = 0; i <= 36; ++i)
			{
				float theta = radians((float)i / segments * angle);
				float thetaPrev = theta - stepRadians;
				float thetaNext = theta + stepRadians;
				float3 p = center + radius * (tangentOS * cos(theta) + bitangentOS * sin(theta));
				float3 pPrev = center + radius * (tangentOS * cos(thetaPrev) + bitangentOS * sin(thetaPrev));
				float3 pNext = center + radius * (tangentOS * cos(thetaNext) + bitangentOS * sin(thetaNext));
				float4 clip = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(p, 1.0f)));
				float4 prevClip = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(pPrev, 1.0f)));
				float4 nextClip = mul(VIEW_PROJECTION_MATRIX, mul(_ModelMatrix, float4(pNext, 1.0f)));
				float2 pixelTangent = (nextClip.xy / nextClip.w - prevClip.xy / prevClip.w) * viewportSize;
				float tangentLengthSq = dot(pixelTangent, pixelTangent);
				pixelTangent *= rsqrt(tangentLengthSq);
				float2 perpendicular = float2(-pixelTangent.y, pixelTangent.x);
				float2 offset = perpendicular * _Thickness * inverseViewportSize * clip.w;

				FragmentVaryings v;
				v.color = color;

				v.positionCS = clip;
				v.positionCS.xy -= offset;
				triangleStream.Append(v);

				v.positionCS = clip;
				v.positionCS.xy += offset;
				triangleStream.Append(v);
			}
			triangleStream.RestartStrip();
		}

		float4 ArcLineFragment(FragmentVaryings input) : SV_TARGET
		{
			return input.color;
		}
		HLSLEND
	}
}