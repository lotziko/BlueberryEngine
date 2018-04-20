package blueberry.engine.UI;

import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;

public class TextureRegionImage extends UiObject {

	private TextureRegion background;
	
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
	
	public void setBackground(TextureRegion background) {
		this.background = background;
	}
	
	@Override
	public void draw(Batch batch) {
		if (isVisible) {
			if (background != null) {
				batch.draw(background, x, y);
			}
		}
	}
	
	public TextureRegionImage(float x, float y, float width, float height, TextureRegion background, int align) {
		super(x, y, width, height, align);
		this.background = background;
	}

}
