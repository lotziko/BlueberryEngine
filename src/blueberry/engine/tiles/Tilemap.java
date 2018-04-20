package blueberry.engine.tiles;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.g2d.Batch;
import com.google.gson.annotations.Expose;

import blueberry.engine.json.JsonManager;

public class Tilemap {
	
	@Expose
	private int width, height, cellWidth, cellHeight;

	@Expose
	private List<Tilelayer> layers;
	
	public int getWidth() {
		return width;
	}

	public int getHeight() {
		return height;
	}
	
	public static void save(String filename, Tilemap tilemap) {
		for (Tilelayer layer : tilemap.layers) {
			layer.replaceTilesWithIDs();
		}
		JsonManager.toJsonFile(filename, tilemap);
	}
	
	public static Tilemap load(String filename, HashMap<Integer, Tile> dictionary) {
		Tilemap tilemap = JsonManager.fromJson(Gdx.files.internal(filename).readString(), Tilemap.class);
		for (Tilelayer layer : tilemap.layers) {
			layer.replaceIDsWithTiles(dictionary);
		}
		return tilemap;
	}
	
	public static Tilemap create(int width, int height, int cellWidth, int cellHeight, Tilelayer... layers) {
		return new Tilemap(width, height, cellWidth, cellHeight, layers);
	}
	
	public void insertTiles(String layer, Tile[][] tiles, int x, int y) {
		
	}
	
	public void insertTiles(Tilemap map, int x, int y) {
		Tilelayer tempLayer;
		for (Tilelayer tilelayer : layers) {
			if ((tempLayer = map.getLayer(tilelayer.name)) != null) {
				for (int i = 0; i < tempLayer.getWidth(); i++) {
					if (x + i < this.width) {
						for (int j = 0; j < tempLayer.getHeight(); j++) {
							if (y + j < this.height) { 
								tilelayer.setTile(x + i, y + j, tempLayer.getTile(i, j));
							} else {
								break;
							}
						}
					} else {
						break;
					}
				}
			}
		}
	}
	
	public void insertTiles(Tilelayer layer, int x, int y) {
		for (Tilelayer tilelayer : layers) {
			if (tilelayer.name.equals(layer.name)) {
				for (int i = 0; i < tilelayer.getWidth(); i++) {
					if (x + i < this.width) {
						for (int j = 0; j < tilelayer.getHeight(); j++) {
							if (y + j < this.height) { 
								tilelayer.setTile(x + i, y + j, layer.getTile(i, j));
							} else {
								break;
							}
						}
					} else {
						break;
					}
				}
			}
		}
	}
	
	public Tilelayer getLayer(String name) {
		for (Tilelayer tilelayer : layers) {
			if (tilelayer.name.equals(name)) {
				return tilelayer;
			}
		}
		return null;
	}
	
	public void draw(Batch batch, float x, float y) {
		for (int i = layers.size() - 1; i >= 0; i--) {
			layers.get(i).draw(batch, x, y);
		}
	}
	
	public void draw(Batch batch, String layer, float x, float y) {
		for (Tilelayer tilelayer : layers) {
			if (tilelayer.name.equals(layer)) {
				tilelayer.draw(batch, x, y);
			}
		}
	}
	
	private Tilemap(int width, int height, int cellWidth, int cellHeight, Tilelayer... layers) {
		this.width = width;
		this.height = height;
		this.cellWidth = cellWidth;
		this.cellHeight = cellHeight;
		this.layers = new ArrayList<>();
		for (Tilelayer tilelayer : layers) {
			this.layers.add(tilelayer);
		}
	}
	
}
