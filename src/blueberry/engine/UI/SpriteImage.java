package blueberry.engine.UI;

import com.badlogic.gdx.graphics.g2d.Batch;

import blueberry.engine.sprites.Sprite;

public class SpriteImage extends UiObject {

	protected Sprite background;
	
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
	
	public float getWidth() {
		return width;
	}

	public void setWidth(float width) {
		this.width = width;
	}

	public float getHeight() {
		return height;
	}

	public void setHeight(float height) {
		this.height = height;
	}
	
	public void setBackground(Sprite background) {
		this.background = background;
	}
	
	@Override
	public void draw(Batch batch) {
		if (isVisible) {
			if (background != null) {
				background.draw(batch, 0, x, y);
			}
		}
	}
	
	public SpriteImage(float x, float y, float width, float height, Sprite background, int align) {
		super(x, y, width, height, align);
		this.background = background;
	}

}
