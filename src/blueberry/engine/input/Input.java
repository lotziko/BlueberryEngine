package blueberry.engine.input;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.InputProcessor;
import com.badlogic.gdx.graphics.Camera;
import com.badlogic.gdx.math.Vector3;

public class Input {

	private static Camera camera, uiCamera;
	/*private static int keyCount = 155;
	private static int mouseButtonCount = 5;
	private static boolean[] keysDown = new boolean[keyCount];
	private static boolean[] keysPressed = new boolean[keyCount];
	private static boolean[] mouseButtonDown = new boolean[mouseButtonCount];
	private static boolean[] mouseButtonPressed = new boolean[mouseButtonCount];*/
	private static int wheel = 0;
	
	
	public static float getMouseX() {
		return camera.unproject(new Vector3(Gdx.input.getX(), Gdx.input.getY(), 0)).x;
	}
	
	public static float getMouseY() {
		return camera.unproject(new Vector3(Gdx.input.getX(), Gdx.input.getY(), 0)).y;
	}
	
	public static float getMouseDeltaX() {
		return uiCamera.unproject(new Vector3(Gdx.input.getDeltaX(), Gdx.input.getDeltaY(), 0)).x;
	}
	
	public static float getMouseDeltaY() {
		return uiCamera.unproject(new Vector3(Gdx.input.getDeltaX(), Gdx.input.getDeltaY(), 0)).y;
	}
	
	public static float getUiMouseX() {
		return uiCamera.unproject(new Vector3(Gdx.input.getX(), Gdx.input.getY(), 0)).x;
	}
	
	public static float getUiMouseY() {
		return uiCamera.unproject(new Vector3(Gdx.input.getX(), Gdx.input.getY(), 0)).y;
	}
	
	public static boolean isKeyDown(int key) {
		return Gdx.input.isKeyPressed(key);
	}
	
	public static boolean isKeyPressed(int key) {
		return Gdx.input.isKeyJustPressed(key);
	}
	
	public static boolean isMouseDown(int key) {
		return Gdx.input.isButtonPressed(key);
	}
	
	public static boolean isMousePressed(int key) {
		return Gdx.input.isButtonPressed(key) && Gdx.input.justTouched();
	}
	
	public static boolean isWheelUp() {
		if (wheel < 0) {
			wheel = 0;
			return true;
		}
		return false;
	}
	
	public static boolean isWheelDown() {
		if (wheel > 0) {
			wheel = 0;
			return true;
		}
		return false;
	}
	
	public static void setCamera(Camera camera, Camera uiCamera) {
		Input.camera = camera;
		Input.uiCamera = uiCamera;
		Gdx.input.setInputProcessor(new InputProcessor() {
			
			@Override
			public boolean touchUp(int arg0, int arg1, int arg2, int arg3) {
				return false;
			}
			
			@Override
			public boolean touchDragged(int arg0, int arg1, int arg2) {
				return false;
			}
			
			@Override
			public boolean touchDown(int arg0, int arg1, int arg2, int arg3) {
				return false;
			}
			
			@Override
			public boolean scrolled(int arg0) {
				wheel = arg0;
				return false;
			}
			
			@Override
			public boolean mouseMoved(int arg0, int arg1) {
				return false;
			}
			
			@Override
			public boolean keyUp(int arg0) {
				return false;
			}
			
			@Override
			public boolean keyTyped(char arg0) {
				return false;
			}
			
			@Override
			public boolean keyDown(int arg0) {
				return false;
			}
		});
	}
}
