package blueberry.engine.world;

import blueberry.engine.UI.UiManager;
import blueberry.engine.math.SimpleMath;
import blueberry.engine.objects.ObjectManager;

public class Room {
	
	private static Room current;
	private static int viewWidth, viewHeight;
	protected static int FADE_IN = 0, FADE_OUT = 1, FADE_PAUSE = 2;
	protected int fadeStatus = FADE_PAUSE;
	protected float fadeAlpha;
	public int width, height;
	public float x, y;
	
	public Room create() {return this;}
	
	public static void step() {
		current.x = SimpleMath.clamp(current.x, viewWidth/2, current.width - viewWidth/2);
		current.y =	SimpleMath.clamp(current.y, viewHeight/2, current.height - viewHeight/2);
		
		if (current != null && current.fadeStatus != FADE_PAUSE) {
			if (current.fadeStatus == FADE_OUT) {
				if (current.fadeAlpha >= 1) {
					current.fadedOut();
				} else {
					current.fadeAlpha += 0.02f;
				}
			} else if (current.fadeStatus == FADE_IN) {
				if (current.fadeAlpha <= 0) {
					current.fadedIn();
				} else {
					current.fadeAlpha -= 0.02f;
				}
			}
		}
	}
	
	public void draw() {};
	
	public Room fadeIn() {
		fadeAlpha = 1f;
		fadeStatus = FADE_IN;
		return this;
	}
	
	public Room fadeOut() {
		fadeStatus = FADE_OUT;
		return this;
	}
	
	/* is being extended */
	
	protected void fadedOut() {
		fadeStatus = FADE_PAUSE;
	}
	
	protected void fadedIn() {
		fadeStatus = FADE_PAUSE;
	}
	
	public static void setViews(int width, int height) {
		viewWidth = width;
		viewHeight = height;
	}
	
	public void setViewPosition(float x, float y) {
		this.x = x;// - viewWidth/2;
		this.y = y;// - viewHeight/2;
	}
	
	public void plusViewPosition(float x, float y) {
		this.x += x;
		this.y += y;
	}
	
	public float getViewPositionX() {
		return x;//; + viewWidth/2;
	}
	
	public float getViewPositionY() {
		return y;// + viewHeight/2;
	}
	
	public float getWorldViewPositionX() {
		return x - viewWidth/2;
	}
	
	public float getWorldViewPositionY() {
		return y - viewHeight/2;
	}
	
	public static Room getCurrentRoom() {
		return current;
	}
	
	public static void setCurrentRoom(Room room) {
		current = room;
	}
	
	public static void gotoRoom(Room room) {
		clear();
		current = room.create();
	}
	
	public static void clear() {
		ObjectManager.clear();
		UiManager.clear();
	}
	
	public Room(int width, int height) {
		this.width = width;
		this.height = height;
	}
	
}
