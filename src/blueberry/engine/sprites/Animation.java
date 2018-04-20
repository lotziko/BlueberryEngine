package blueberry.engine.sprites;

import java.util.ArrayList;
import java.util.List;

public class Animation {

	private static List<Animation> animations = new ArrayList<>();
	private boolean isPaused;
	private float speed, currentFrame;

	private int size;
	
	public static void step() {
		for (Animation animation : animations) {
			animation.stepPrivate();
		}
	}
	
	private void stepPrivate() {
		if (!isPaused) {
			if (speed > 0) {
				if (currentFrame + speed < size) {
					currentFrame += speed;
				} else {
					currentFrame = 0;
					end();
				}
			} else {
				if (currentFrame + speed > 0) {
					currentFrame += speed;
				} else {
					currentFrame = size;
					end();
				}
			}
		}
	}
	
	public boolean isPaused() {
		return isPaused;
	}
	
	public void pause() {
		isPaused = true;
	}
	
	public void resume() {
		isPaused = false;
	}
	
	public void setSpeed(float speed) {
		this.speed = speed;
	}
	
	public float getSpeed() {
		return speed;
	}
	
	public void setSize(int size) {
		this.size = size;
	}
	
	public int getSize() {
		return size;
	}
	
	public float getCurrentFrame() {
		return currentFrame;
	}

	public void setCurrentFrame(float currentFrame) {
		this.currentFrame = currentFrame;
	}
	
	public void end() {}
	
	public void dispose() {
		animations.remove(this);
	}
	
	public Animation(float speed, int size) {
		this.speed = speed;
		this.size = size;
		animations.add(this);
	}
	
}
