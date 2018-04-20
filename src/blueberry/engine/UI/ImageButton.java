package blueberry.engine.UI;

import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.NinePatch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;

public class ImageButton extends Button {

	private float imagePosX, imagePosY;
	private TextureRegion image;
	
	@Override
	public void setPosition(float x, float y) {
		super.setPosition(x, y);
		if (image != null) {
			imagePosX = (x + (width - image.getRegionWidth())/2);
			imagePosY = (y + (height - image.getRegionHeight())/2);
		}
	}
	
	@Override
	public void draw(Batch batch) {
		super.draw(batch);
		if (isVisible && image != null) {
			batch.draw(image, imagePosX, imagePosY);
		}
	}
	
	public ImageButton(float x, float y, float width, float height, NinePatch background, NinePatch hover, TextureRegion image, int align) {
		super(x, y, width, height, background, hover, align);
		this.image = image;
		setPosition(x, y);
	}

}