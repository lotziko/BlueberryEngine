package blueberry.engine.tiles;

import java.util.HashMap;

import com.badlogic.gdx.graphics.g2d.Batch;
import com.google.gson.annotations.Expose;


public class Tilelayer {
	
	@Expose
	private int[][] tileIDs;
	@Expose
	String name;
	private Tile[][] tileObjects;
	@Expose
	private int width, height, cellWidth, cellHeight;
	
	public int getWidth() {
		return width;
	}
	
	public int getHeight() {
		return height;
	}
	
	public int getCellWidth() {
		return cellWidth;
	}

	public int getCellHeight() {
		return cellHeight;
	}
	
	public Tile getTile(int x, int y) {
		if (x >= 0 && x < width && y >= 0 && y < height) {
			return tileObjects[x][y];
		}
		return null;
	}
	
	public void setTile(int x, int y, Tile tile) {
		if (x >= 0 && x < width && y >= 0 && y < height) {
			tileObjects[x][y] = tile;
		}
	}
	
	public void fillRectangle(int x1, int y1, int x2, int y2, Tile tile) {
		for(int i = Math.max(0, x1); i < Math.min(width, x2); i++) {
			for(int j = Math.max(0, y1); j < Math.min(height, y2); j++) {
				tileObjects[i][j] = tile;
			}
		}
	}

	public void resize(int newWidth, int newHeight) {
		Tile[][] newObjects = new Tile[newWidth][newHeight];
		for(int i = 0; i < newWidth; i++) {
			for(int j = 0; j < newHeight; j++) {
				newObjects[i][j] = tileObjects[Math.floorDiv(width * i, newWidth)][Math.floorDiv(height * j, newHeight)];
			}
		}
		tileObjects = newObjects;
		this.width = newWidth;
		this.height = newHeight;
	}
	
	public void draw(Batch batch, float x, float y) {
		Tile tile;
		for(int i = 0; i < width; i++) {
			for(int j = 0; j < height; j++) {
				if ((tile = tileObjects[i][j]) != null) {
					tile.draw(batch, x + i * cellWidth, y + j * cellHeight - tile.getImage().getRegionHeight() + 8);
				}
			}
		}
	}
	
	public void replaceIDsWithTiles(HashMap<Integer, Tile> dictionary) {
		tileObjects = new Tile[width][height];
		for(int i = 0; i < width; i++) {
			for(int j = 0; j < height; j++) {
				tileObjects[i][j] = dictionary.get(tileIDs[i][j]);
			}
		}
	}
	
	public void replaceTilesWithIDs() {
		tileIDs = new int[width][height];
		for(int i = 0; i < width; i++) {
			for(int j = 0; j < height; j++) {
				if (tileObjects[i][j] != null)
				tileIDs[i][j] = tileObjects[i][j].getID();
			}
		}
	}
	
	public boolean pointCollide(int x, int y) {
		if ((x = Math.floorDiv(x, cellWidth)) > 0 && (y = Math.floorDiv(y, cellHeight)) > 0 && x < width && y < height)
		if (tileObjects[x][y] != null) {
			return true;
		}
		return false;
	}
	
	public boolean pointCollide(double x, double y) {
		int x1, y1;
		if ((x1 = Math.floorDiv((int)x, cellWidth)) > 0 && (y1 = Math.floorDiv((int)y, cellHeight)) > 0 && x1 < width && y1 < height)
		if (tileObjects[x1][y1] != null) {
			return true;
		}
		return false;
	}
	
	/* Can be slow */
	
	public boolean lineCollide(int x1, int y1, int x2, int y2) {
		double angleRadians = Math.atan2(y2 - y1, x2 - x1);
		double xOffset = Math.cos(angleRadians) * cellWidth;
		double yOffset = Math.sin(angleRadians) * cellHeight;
		if (xOffset > 0) {
			double x, y;
			for(x = x1, y = y2; x < x2; x += xOffset, y += yOffset) {
				if (pointCollide(x, y)) {
					return true;
				}
			}
		} else {
			double x, y;
			for(x = x1, y = y2; x > x2; x += xOffset, y += yOffset) {
				if (pointCollide(x, y)) {
					return true;
				}
			}
		}
		
		return false;
	}
	
	public Tilelayer(String name, int width, int height, int cellWidth, int cellHeight, Tile[][] tileObjects) {
		this.name = name;
		this.width = width;
		this.height = height;
		this.cellWidth = cellWidth;
		this.cellHeight = cellHeight;
		if (tileObjects != null) {
			this.tileObjects = tileObjects;
		} else {
			this.tileObjects = new Tile[width][height];
		}
	}
	
}
