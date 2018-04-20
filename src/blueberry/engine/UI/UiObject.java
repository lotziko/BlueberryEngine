package blueberry.engine.UI;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer.ShapeType;

public class UiObject {
	
	protected int align;
	protected float x, y, width, height;
	protected boolean isVisible = true;
	protected UiObject parent;
	protected InputListener listener;
	
	public UiObject getParent() {
		return parent;
	}
	
	public InputListener getListener() {
		return listener;
	}
	
	public void setListener(InputListener listener) {
		if (this.listener != null) {
			this.listener.removeListener();
		}
		this.listener = listener;
		if (listener != null) {
			this.listener.x = x;
			this.listener.y = y;
			this.listener.width = width;
			this.listener.height = height;
		}
	}
	
	public void setPosition(float x, float y) {
		switch (align) {
		case Align.NULL:
			this.x = x;
			this.y = y;
			break;
		case Align.TOP_CENTER:
			this.x = x - width/2;
			this.y = y;
			break;
		case Align.CENTER:
			this.x = x - width/2;
			this.y = y - height/2;
			break;
		case Align.RIGHT:
			this.x = x - width;
			this.y = y;
		default:
			break;
		}
		if (listener != null) {
			listener.x = this.x;
			listener.y = this.y;
		}
	}
	
	public void setSize(float width, float height) {
		this.width = width;
		this.height = height;
	}
	
	public void setVisible(boolean isVisible) {
		this.isVisible = isVisible;
		if (listener != null) {
			listener.isVisible = isVisible;
		}
	}
	
	public boolean isVisible() {
		return isVisible;
	}
	
	public void dispose(){}
	
	public void draw(Batch batch) {
		ShapeRenderer render = new ShapeRenderer();
		render.setProjectionMatrix(batch.getProjectionMatrix());
		render.setTransformMatrix(batch.getTransformMatrix());
		batch.end();
		render.begin(ShapeType.Line);
		render.rect(x, y, width, height);
		if (listener != null) {
			render.setColor(Color.RED);
			render.rect(x, y, width, height);
			render.setColor(Color.WHITE);
		}
		render.end();
		batch.begin();
	}
	
	public UiObject(float x, float y, float width, float height, int align) {
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.align = align;
	}
	
	public class Align {
		public static final int NULL = -1, LEFT = 0, CENTER = 1, TOP_CENTER = 2, RIGHT = 3;
	}
}
