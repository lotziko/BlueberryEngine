package blueberry.engine.UI;

import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.g2d.Batch;

public class UiManager {
	public static Table stage;
	private static Batch batch;
	public static OrthographicCamera camera;
	
	public static void setBatch(Batch batchToSet) {
		batch = batchToSet;
	}
	
	public static void setCamera(OrthographicCamera camera) {
		UiManager.camera = camera;
	}
	
	public static void drawAll() {
		if (stage != null) {
			stage.draw(batch);
		}
	}
	
	public static void clear() {
		stage = null;
		InputListener.listeners.clear();
	}
	
}
