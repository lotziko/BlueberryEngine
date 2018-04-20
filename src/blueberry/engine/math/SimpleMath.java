package blueberry.engine.math;

public class SimpleMath {

	public static float clamp(float value, int min, int max) {
		if (value < min) {
			return min;
		} else if (value > max) {
			return max;
		}
		return value;
	}
	
	public static int clamp(int value, int min, int max) {
		if (value < min) {
			return min;
		} else if (value > max) {
			return max;
		}
		return value;
	}
	
	public static float clamp(float value, float min, float max) {
		if (value < min) {
			return min;
		} else if (value > max) {
			return max;
		}
		return value;
	}
	
	/**
	 * direction in degrees
	 * @param distance
	 * @param direction
	 * @return
	 */
	
	public static double lengthdirX(float distance, float direction) {
		return Math.cos(Math.toRadians(direction)) * distance;
	}
	
	public static double lengthdirY(float distance, float direction) {
		return Math.sin(Math.toRadians(direction)) * distance;
	}
	
	public static double distance(float x1, float y1, float x2, float y2) {
		return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
	}
	
	public static double direction(float x1, float y1, float x2, float y2) {
		return Math.atan2((y2 - y1), (x2 - x1));
	}
	
	public static float sign(Float num) {
		return num.compareTo(0f);
	}
	
	public static float lerp(float num1, float num2, float alpha)
	{
	    return num1 + alpha * (num2 - num1);
	}
	
}
