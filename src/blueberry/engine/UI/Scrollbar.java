package blueberry.engine.UI;

import com.badlogic.gdx.Input.Buttons;
import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.NinePatch;

import blueberry.engine.input.Input;
import blueberry.engine.math.SimpleMath;

public class Scrollbar extends Table {

	private Button scroll;
	private float scrollYtarget;
	private NinePatch background;
	
	public float getYoffset() {
		return y - scroll.y;
	}
	
	@Override
	public void setPosition(float x, float y) {
		super.setPosition(x, y);
		scrollYtarget = scroll.y;
	}
	
	public void setScrollHeight(float height) {
		if (height < this.height) {
			scroll.y = y;
		}
		scroll.height = SimpleMath.clamp(this.height/ (height / this.height), 0, this.height);
		scroll.listener.height = scroll.height;
	}
	
	@Override
	public void draw(Batch batch) {
		if (background != null) {
			background.draw(batch, x, y, width, height);
		}
		super.draw(batch);
	}

	public void setStyle(NinePatch buttonBackground, NinePatch buttonHover, NinePatch background) {
		scroll.background = buttonBackground;
		scroll.hover = buttonHover;
		this.background = background;
	}
	
	public Scrollbar(float x, float y, float width, float height, int align) {
		super(x, y, width, height, align);
		scroll = new Button(x, y, width, SimpleMath.clamp(20f, 0f, height), null, null, Align.NULL) {

			@Override
			public void draw(Batch batch) {
				if (Math.abs(scrollYtarget - y) > 0.05f) {
					setPosition(x, y + (scrollYtarget - y)/20);
				}
				super.draw(batch);
			}
			
		};
		scroll.setListener(new InputListener(Buttons.LEFT) {

			private float mousePreviousY;
			
			@Override
			public void released() {
				
			}
			
			@Override
			public void entered() {
				mousePreviousY = Input.getUiMouseY();
			}
			
			@Override
			public void click() {
				mousePreviousY = Input.getUiMouseY();
			}
			
			@Override
			public void hold() {
				float mouseY = Input.getUiMouseY(), yDelta = mouseY - mousePreviousY;
				scroll.setPosition(x, SimpleMath.clamp(scroll.y + yDelta, Scrollbar.this.y, Scrollbar.this.y + Scrollbar.this.height - scroll.height));
				scrollYtarget = scroll.y;
				mousePreviousY = mouseY;
				InputListener.focused = this;
				super.hold();
			}
			
		});
		this.addChildren(scroll);
		
		this.setListener(new InputListener(Buttons.LEFT) {

			@Override
			public void click() {
				scrollYtarget = SimpleMath.clamp(Input.getUiMouseY() - scroll.height/2, Scrollbar.this.y, Scrollbar.this.y + Scrollbar.this.height - scroll.height);
				super.click();
			}
			
		});
	}

}
