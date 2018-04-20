package blueberry.engine.UI;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input.Buttons;

import blueberry.engine.input.Input;

public class InputListener {

	static List<InputListener> listeners = new CopyOnWriteArrayList<>();

	/* User not have to touch this */

	float x, y, width, height;
	public boolean isHovered = false, isPressed = false, isHold = false, isVisible = true;
	int gamepadShortcut = 0;
	private int button;

	private static float mouseX, mouseY;
	public static InputListener focused;

	public static void step() {
		mouseX = Input.getUiMouseX();
		mouseY = Input.getUiMouseY();
		boolean hoveredFound = false, clicked = false;
		if (focused == null) {
			for (InputListener clickListener : listeners) {
				if (clickListener.isVisible && !hoveredFound && mouseX > clickListener.x && mouseX < clickListener.x + clickListener.width && mouseY > clickListener.y && mouseY < clickListener.y + clickListener.height) {
					if (!clickListener.isHovered) {
						clickListener.entered();
					}
					clickListener.isHovered = true;
					clickListener.hover();
					if (Gdx.input.isButtonPressed(clickListener.button)) {
						if (Gdx.input.justTouched()) {
							clickListener.isHold = true;
							clickListener.isPressed = true;
							clickListener.click();
						} else {
							clickListener.isPressed = false;
						}
						if (clickListener.isHold) {
							clickListener.hold();
						}
					} else {
						clickListener.isHold = false;
						clickListener.isPressed = false;
					}
					hoveredFound = true;
				} else {
					clickListener.isHovered = false;
				}
			}
			clicked = Gdx.input.isButtonPressed(Buttons.LEFT);
			if (!hoveredFound && clicked) {
				for (InputListener clickListener : listeners) {
					clickListener.noTouchClick();
				}
			}
		}
		else {
			if (!Gdx.input.isButtonPressed(focused.button)) {
				focused.released();
				focused = null;
			} else {
				focused.hold();
			}
		}
	}

	public void click() {
	}

	public void hover() {
	}

	public void hold() {
	}

	public void noTouchClick() {
	}

	public void entered() {
	}

	public void released() {
	}

	public InputListener(int button) {
		listeners.add(this);
	}

	public void removeListener() {
		listeners.remove(this);
	}
}
