package blueberry.engine.UI;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.math.Rectangle;
import com.badlogic.gdx.scenes.scene2d.utils.ScissorStack;

import blueberry.engine.world.Room;

public class Table extends UiObject {

	protected List<UiObject> children = new CopyOnWriteArrayList<>();
	protected Rectangle bounds;
	protected Rectangle scissors = new Rectangle();
	protected float origX, origY;
	protected int space;
	protected ResizeListener resizeListener;
	protected boolean isResiziable = true;
	
	public void setResiziable(boolean isResiziable) {
		this.isResiziable = isResiziable;
	}
	
	@Override 
	public void setSize(float width, float height) {
		super.setSize(width, height);
		if (resizeListener != null) {
			resizeListener.resized(width, height);
		}
	}
	
	public void setSpace(int space) {
		this.space = space;
	}
	
	@Override
	public void setVisible(boolean isVisible) {
		super.setVisible(isVisible);
		for (UiObject uiObject : children) {
			uiObject.setVisible(isVisible);
		}
	}
	
	@Override
	public void draw(Batch batch) {
		origX = bounds.x;
		origY = bounds.y;
		bounds.x = Room.getCurrentRoom().getWorldViewPositionX() + origX;
		bounds.y = Room.getCurrentRoom().getWorldViewPositionY() + origY;
		ScissorStack.calculateScissors(UiManager.camera, batch.getTransformMatrix(), bounds, scissors);
		batch.flush();
		if (ScissorStack.pushScissors(scissors)) {
			for (UiObject uiObject : children) {
				uiObject.draw(batch);
			}
			batch.flush();
			ScissorStack.popScissors();
		}
		bounds.x = origX;
		bounds.y = origY;
		super.draw(batch);
	}
	
	protected void layout() {
		if (align != Align.NULL) {
			int height = 0, totalHeight = 0;
			for (int i = 0; i < children.size(); i++) {
				height += children.get(i).height;
				if (i < children.size() - 1) {
					height += space;
				}
			}
			float y = this.y;// + (this.height - height) / 2;
			if (align == Align.CENTER) {
				y += (this.height - height) / 2;
			}
			for (UiObject uiObject : children) {
				float xOffset = 0, yOffset = 0;
				if (uiObject.align == Align.TOP_CENTER) {
					xOffset = width/2;
				}
				uiObject.setPosition(this.x + xOffset/* + this.width / 2*/, y);
				y += uiObject.height + space;
				totalHeight += uiObject.height + space;
			}
			if (isResiziable)
			setSize(width, totalHeight);
		}
	}
	
	public void childrenResized() {}
	
	@Override
	public void setPosition(float x, float y) {
		float oldX = this.x, oldY = this.y;
		super.setPosition(x, y);
		float offsetX = this.x - oldX, offsetY = this.y - oldY;
		
		for (UiObject uiObject : children) {
			uiObject.setPosition(uiObject.x + offsetX, uiObject.y + offsetY);
		}
		bounds = new Rectangle(this.x, this.y, width, height);
	}

	public void addChildren(UiObject children) {
		this.children.add(children);
		children.parent = this;
		children.setVisible(isVisible);
		layout();
	}
	
	public void clearChildren() {
		for (UiObject uiObject : children) {
			uiObject.dispose();
			uiObject.parent = null;
			uiObject = null;
		}
		children.clear();
	}
	
	public List<UiObject> getChildren() {
		return children;
	}
	
	public void removeChildren(UiObject children) {
		children.dispose();
		children.parent = null;
		this.children.remove(children);
		children = null;
		layout();
	}
	
	public Table(float x, float y, float width, float height, int align) {
		super(x, y, width, height, align);
		bounds = new Rectangle(x, y, width, height);
		setSize(width, height);
	}
	
}
