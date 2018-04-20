package blueberry.engine.UI;

import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.NinePatch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;

import blueberry.engine.math.SimpleMath;

public class ProgressBar extends UiObject {

	private NinePatch background;
	private TextureRegion texture;
	private int maxSize, currentSize;
	
	public int getMaxSize() {
		return maxSize;
	}

	public void setMaxSize(int maxSize) {
		this.maxSize = maxSize;
	}

	public int getCurrentSize() {
		return currentSize;
	}

	public void setCurrentSize(int currentSize) {
		this.currentSize = (int)SimpleMath.clamp(currentSize, 0, maxSize);
	}

	@Override
	public void draw(Batch batch) {
		if (isVisible) {
			background.draw(batch, x, y, width, height);
			for(int i = 0; i < currentSize; i++) {
				batch.draw(texture, x + i * texture.getRegionWidth() + background.getLeftWidth(), y + background.getTopHeight());
			}
		}
		super.draw(batch);
	}

	public ProgressBar(float x, float y, float width, float height, NinePatch background, TextureRegion texture, int align) {
		super(x, y, width, height, align);
		this.background = background;
		this.texture = texture;
	}

}
