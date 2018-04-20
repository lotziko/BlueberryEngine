package blueberry.engine.tiles;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;

public class Tile {
	private int ID;
	private TextureRegion image;
	private String tileset;
	
	public String getTileset() {
		return tileset;
	}
	
	public int getID() {
		return ID;
	}
	
	public TextureRegion getImage() {
		return image;
	}
	
	public void draw(Batch batch, float x, float y) {
		batch.draw(image, x, y);
	}
	
	public void draw(Batch batch, float x, float y, float alpha) {
		Color color = batch.getColor();
		Color alphaColor = new Color(color);
		alphaColor.a = alpha;
		batch.setColor(alpha);
		batch.draw(image, x, y);
		batch.setColor(color);
	}
	
	public Tile(int ID, TextureRegion image) {
		this.ID = ID;
		this.image = image;
	}
	
	public Tile(int ID, TextureRegion image, String tileset) {
		this(ID, image);
		this.tileset = tileset;
	}

	
}
