package blueberry.engine.objects;

import blueberry.engine.sprites.Sprite;

public class Object {

	protected int ID, depth;
	protected float x, y;
	protected Sprite sprite;
	
	public int getID() {
		return ID;
	}

	public void setID(int iD) {
		ID = iD;
	}

	public float getX() {
		return x;
	}

	public void setX(float x) {
		this.x = x;
	}

	public float getY() {
		return y;
	}

	public void setY(float y) {
		this.y = y;
	}
	
	public int getDepth() {
		return depth;
	}

	public void setDepth(int depth) {
		this.depth = depth;
	}
	
	public void setPosition(float x, float y) {
		this.x = x;
		this.y = y;
	}
	
	public void move(float xDelta, float yDelta) {}
	
	public void step(){};
	
	public void drawBegin(){};
	
	public void draw(){};
	
	public void drawEnd(){};
	
	public void drawGUI(){};
	
	public void destroy(){};
	
	public void create(){};
	
}
