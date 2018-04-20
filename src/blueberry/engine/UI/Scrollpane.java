package blueberry.engine.UI;

import com.badlogic.gdx.graphics.g2d.Batch;

public class Scrollpane extends Table {

	public Scrollbar scrollbar;
	private Table children;
	
	@Override
	public void draw(Batch batch) {
		if (children != null)
		children.setPosition(x, y + (scrollbar.getYoffset()));
		super.draw(batch);
	}
	
	public void setChildren(Table children) {
		this.children = children;
		children.resizeListener = new ResizeListener() {

			@Override
			public void resized(float width, float height) {
				scrollbar.setScrollHeight(height);
				super.resized(width, height);
			}
			
		};
		children.setSize(children.width, children.height);
		addChildren(children);
	}
	
	public Scrollpane(float x, float y, float width, float height) {
		super(x, y, width, height, Align.NULL);
		scrollbar = new Scrollbar(x + width - 5, y, 5, height, Align.NULL) {
			
		};
		addChildren(scrollbar);
	}
}
