#include "MathHelper.h"

namespace Blueberry
{
	inline bool IsInsideRectangle(const Vector2& position, const Vector4& rectangle)
	{
		return (position.x >= rectangle.x && position.x <= rectangle.z && position.y >= rectangle.y && position.y <= rectangle.w);
	}

	inline float Sign(const Vector2& p1, const Vector2& p2, const Vector2& p3)
	{
		return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
	}

	inline bool IsInsideTriangle(const Vector2& position, const Vector2& p1, const Vector2& p2, const Vector2& p3)
	{
		bool b1 = Sign(position, p1, p2) < 0.0f;
		bool b2 = Sign(position, p2, p3) < 0.0f;
		bool b3 = Sign(position, p3, p1) < 0.0f;

		return (b1 == b2) && (b2 == b3);
	}

	inline float Orient2D(const Vector2& p1, const Vector2& p2, const Vector2& p3)
	{
		return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
	}

	inline bool SegmentsIntersect(const Vector2& p1, const Vector2& p2, const Vector2& q1, const Vector2& q2)
	{
		float o1 = Orient2D(p1, p2, q1);
		float o2 = Orient2D(p1, p2, q2);
		float o3 = Orient2D(q1, q2, p1);
		float o4 = Orient2D(q1, q2, p2);

		if ((o1 * o2 < 0) && (o3 * o4 < 0))
		{
			return true;
		}

		return false;
	}

	bool MathHelper::Intersects(const Vector4& rectangle, const Vector2& p1, const Vector2& p2, const Vector2& p3)
	{
		// Triangle vertex in rectangle
		if (IsInsideRectangle(p1, rectangle))
		{
			return true;
		}
		if (IsInsideRectangle(p2, rectangle))
		{
			return true;
		}
		if (IsInsideRectangle(p3, rectangle))
		{
			return true;
		}

		Vector2 r1 = Vector2(rectangle.x, rectangle.y);
		Vector2 r2 = Vector2(rectangle.z, rectangle.y);
		Vector2 r3 = Vector2(rectangle.x, rectangle.w);
		Vector2 r4 = Vector2(rectangle.z, rectangle.w);

		// Rectangle vertex in triangle
		if (IsInsideTriangle(r1, p1, p2, p3))
		{
			return true;
		}
		if (IsInsideTriangle(r2, p1, p2, p3))
		{
			return true;
		}
		if (IsInsideTriangle(r3, p1, p2, p3))
		{
			return true;
		}
		if (IsInsideTriangle(r4, p1, p2, p3))
		{
			return true;
		}

		// Segments
		if (SegmentsIntersect(p1, p2, r1, r2))
		{
			return true;
		}
		if (SegmentsIntersect(p1, p2, r2, r3))
		{
			return true;
		}
		if (SegmentsIntersect(p1, p2, r3, r4))
		{
			return true;
		}
		if (SegmentsIntersect(p1, p2, r4, r1))
		{
			return true;
		}

		if (SegmentsIntersect(p2, p3, r1, r2))
		{
			return true;
		}
		if (SegmentsIntersect(p2, p3, r2, r3))
		{
			return true;
		}
		if (SegmentsIntersect(p2, p3, r3, r4))
		{
			return true;
		}
		if (SegmentsIntersect(p2, p3, r4, r1))
		{
			return true;
		}

		if (SegmentsIntersect(p3, p1, r1, r2))
		{
			return true;
		}
		if (SegmentsIntersect(p3, p1, r2, r3))
		{
			return true;
		}
		if (SegmentsIntersect(p3, p1, r3, r4))
		{
			return true;
		}
		if (SegmentsIntersect(p3, p1, r4, r1))
		{
			return true;
		}

		return false;
	}
}