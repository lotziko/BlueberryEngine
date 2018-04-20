package blueberry.engine.random;

import java.util.Random;

public class Rand {

	private static Random random = new java.util.Random();

	public static void setSeed(long seed) {
		random.setSeed(seed);
	}
	
	public static void randomize() {
		random.setSeed(System.currentTimeMillis());
	}
	
	public static int getRandom(int max) {
		return max == 0 ? 0 : random.nextInt(max);
	}
	
	public static int choose(int... ints) {
		return ints[getRandom(ints.length)];
	}
	
	public static int getRandomRange(int min, int max) {
		return random.nextInt(max - min) + min;
	}

	public static float getRandomRange(float min, float max) {
		return min + random.nextFloat() * (max - min);
	}

}
