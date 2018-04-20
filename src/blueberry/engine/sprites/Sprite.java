package blueberry.engine.sprites;

import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.TextureAtlas.AtlasRegion;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.utils.Array;

public class Sprite {

	private String name;
	private int frameCount, xOrigin, yOrigin, width, height;
	private Vector2 onePixel;
	private TextureRegion[] frames;

	public String getName() {
		return name;
	}

	public Vector2 getOnePixel() {
		return onePixel;
	}
	
	public int getFrameCount() {
		return frameCount;
	}

	public int getxOrigin() {
		return xOrigin;
	}

	public int getyOrigin() {
		return yOrigin;
	}

	public TextureRegion[] getFrames() {
		return frames;
	}

	public Sprite(int xOrigin, int yOrigin, String name, Array<AtlasRegion> frames) {
		this.xOrigin = xOrigin;
		this.yOrigin = yOrigin;
		this.frames = new TextureRegion[frames.size];
		for(int i = 0; i < frames.size; i++) {
			this.frames[i] = frames.get(i); 
		}
		this.name = name;
		if (frames != null && this.frames.length > 0) {
			this.width = this.frames[0].getRegionWidth();
			this.height = this.frames[0].getRegionHeight();
			onePixel = new Vector2(1f / this.frames[0].getTexture().getWidth(), 1f/ this.frames[0].getTexture().getHeight());
		}
		frameCount = this.frames.length;
		for(int i = 0; i < frameCount; i++) {
			this.frames[i].flip(false, true);
		}
	}

	public void draw(Batch batch, int index, float x, float y) {
		if (index >= 0 && index < frameCount)
			batch.draw(frames[index], x - xOrigin, y - yOrigin);
	}

	public void draw(Batch batch, int index, float x, float y, float angle) {
		if (index >= 0 && index < frameCount)
			batch.draw(frames[index], x - xOrigin, y - yOrigin, xOrigin, yOrigin, width, height, 1, 1, angle);
	}

	public void draw(Batch batch, int index, float x, float y, float width, float height) {
		if (index >= 0 && index < frameCount)
			batch.draw(frames[index], x - xOrigin, y - yOrigin, width, height);
	}
	
	public void drawScale(Batch batch, int index, float x, float y, float xScale, float yScale) {
		if (index >= 0 && index < frameCount)
			batch.draw(frames[index], x - xOrigin, y - yOrigin, xOrigin, yOrigin, frames[index].getRegionWidth(), frames[index].getRegionHeight(), xScale, yScale, 0f);
	}

	/* TODO: Other methods */

}
