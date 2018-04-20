package blueberry.engine.UI;

import com.badlogic.gdx.Input.Buttons;
import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.NinePatch;

public class Button extends UiObject {

	protected NinePatch background, hover;
	private float alpha = 1f;
	
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
	
	public float getAlpha() {
		return alpha;
	}

	public void setAlpha(float alpha) {
		this.alpha = alpha;
	}
	
	public void setBackground(NinePatch background) {
		this.background = background;
	}
	
	public void setHover(NinePatch hover) {
		this.hover = hover;
	}
	
	@Override
	public void draw(Batch batch) {
		if (isVisible) {
			//Color color = batch.getColor();
			//Color alphaColor = new Color(color);
			//alphaColor.a = alpha;
			//batch.setColor(alphaColor);
			if (background != null) {
				background.draw(batch, x, y, width, height);
			}
			if (listener != null) {
				if (listener.isHovered && hover != null) {
					hover.draw(batch, x, y, width, height);
				}
			}
			//batch.setColor(color);
		}
		super.draw(batch);
	}
	
	@Override
	public void dispose() {
		if (listener != null) {
			listener.removeListener();
			listener = null;
		}
		super.dispose();
	}
	
	public Button(float x, float y, float width, float height, NinePatch background, NinePatch hover, int align) {
		super(x, y, width, height, align);
		this.background = background;
		this.hover = hover;
		setListener(new InputListener(Buttons.LEFT));
	}

}
