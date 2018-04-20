package blueberry.engine.UI;

import com.badlogic.gdx.graphics.g2d.Batch;

public class ButtonHoverTable extends Table {

	InputListener lastHovered;
	
	@Override
	public void draw(Batch batch) {
		for (UiObject uiObject : children) {
			if (uiObject instanceof Button)
			if (((Button)uiObject).getListener().isHovered) {
				lastHovered = ((Button)uiObject).getListener();
			}
		}
		if (lastHovered != null) {
			lastHovered.isHovered = true;
		}
		super.draw(batch);
	}

	public ButtonHoverTable(int x, int y, int width, int height, int align) {
		super(x, y, width, height, align);
		
	}
	
}
