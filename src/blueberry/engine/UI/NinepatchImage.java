package blueberry.engine.UI;

import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.NinePatch;

public class NinepatchImage extends UiObject {

	private NinePatch background;
	
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
	
	public void setBackground(NinePatch background) {
		this.background = background;
	}
	
	@Override
	public void draw(Batch batch) {
		if (isVisible) {
			if (background != null) {
				background.draw(batch, x, y, width, height);
			}
		}
	}
	
	public NinepatchImage(float x, float y, float width, float height, NinePatch background, int align) {
		super(x, y, width, height, align);
		this.background = background;
	}

}
