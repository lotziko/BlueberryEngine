package blueberry.engine.render;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.graphics.glutils.FrameBuffer;
import com.badlogic.gdx.graphics.glutils.ShaderProgram;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer.ShapeType;
import com.badlogic.gdx.scenes.scene2d.utils.TiledDrawable;

import blueberry.engine.math.SimpleMath;
import blueberry.engine.sprites.Sprite;
import blueberry.engine.tiles.Tile;
import blueberry.engine.tiles.Tilelayer;
import blueberry.engine.tiles.Tilemap;

public class Render {

	private Batch batch;
	private ShaderProgram defaultShader;
	private ShapeRenderer shapeRenderer;
	private FrameBuffer buffer;
	private Buffer tempBuffer;
	private OrthographicCamera camera;
	private BitmapFont defaultFont;

	public ShaderProgram getDefaultShader() {
		return defaultShader;
	}
	
	public void setDefaultFont(BitmapFont defaultFont) {
		this.defaultFont = defaultFont;
	}
	
	public BitmapFont getDefaultFont() {
		return defaultFont;
	}
	
	public void drawSprite(Sprite sprite, int index, float x, float y) {
		sprite.draw(batch, index, x, y);
	}
	
	public void drawSprite(Batch batch, Sprite sprite, int index, float x, float y) {
		sprite.draw(batch, index, x, y);
	}

	public void drawSprite(Sprite sprite, int index, float x, float y, float angle) {
		sprite.draw(batch, index, x, y, angle);
	}
	
	public void drawSprite(Batch batch, Sprite sprite, int index, float x, float y, float angle) {
		sprite.draw(batch, index, x, y, angle);
	}

	public void drawSprite(Sprite sprite, int index, float x, float y, float width, float height) {
		sprite.draw(batch, index, x, y, width, height);
	}
	
	public void drawSpriteScale(Sprite sprite, int index, float x, float y, float xScale, float yScale) {
		sprite.drawScale(batch, index, x, y, xScale, yScale);
	}

	public void drawTile(Tile tile, float x, float y) {
		tile.draw(batch, x, y);
	}

	public void drawTile(Tile tile, float x, float y, float alpha) {
		tile.draw(batch, x, y, alpha);
	}

	public void drawTilemap(Tilemap tilemap, float x, float y) {
		tilemap.draw(batch, x, y);
	}
	
	public void drawTilelayer(Tilelayer tilelayer, float x, float y) {
		tilelayer.draw(batch, x, y);
	}
	
	public void drawText(float x, float y, String text) {
		defaultFont.getData().setScale(1, -1);
		defaultFont.draw(batch, text, x, y);
		defaultFont.getData().setScale(1, 1);
	}
	
	public void drawText(Batch batch, float x, float y, String text) {
		defaultFont.getData().setScale(1, -1);
		defaultFont.draw(batch, text, x, y);
		defaultFont.getData().setScale(1, 1);
	}
	
	public void drawTextureRegion(TextureRegion region, float x, float y, float width, float height) {
		batch.draw(region, x, y, width, height);
	}
	
	public void drawRectangle(int x, int y, int width, int height) {
		batch.end();
		
		shapeRenderer.begin(ShapeType.Filled);
		shapeRenderer.rect(x, y, width, height);
		shapeRenderer.end();
		
		batch.begin();
	}
	
	public void drawRectangle(float x, float y, float width, float height) {
		batch.end();
		
		shapeRenderer.begin(ShapeType.Line);
		shapeRenderer.rect(x, y, width, height);
		shapeRenderer.end();
		
		batch.begin();
	}
	
	public void drawBackground(int x, int y, int width, int height, Color color) {
		batch.end();
		Color clr = new Color(shapeRenderer.getColor());
		shapeRenderer.setColor(color);
		shapeRenderer.begin(ShapeType.Filled);
		shapeRenderer.rect(x, y, width, height);
		shapeRenderer.end();
		shapeRenderer.setColor(clr);
		batch.begin();
	}
	
	public void drawRectangle(float x, float y, float width, float height, Color color) {
		batch.end();
		Gdx.gl.glBlendFunc(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA);
		Gdx.gl.glEnable(GL20.GL_BLEND);
		Color clr = shapeRenderer.getColor();
		shapeRenderer.setColor(color);
		shapeRenderer.begin(ShapeType.Filled);
		shapeRenderer.rect(x, y, width, height);
		shapeRenderer.end();
		shapeRenderer.setColor(clr);
		Gdx.gl.glDisable(GL20.GL_BLEND);
		batch.begin();
	}
	
	public void setShader(ShaderProgram shader) {
		batch.setShader(shader);
	}
	
	public void resetShader() {
		batch.setShader(defaultShader);
	}

	public Buffer bufferCreate(int width, int height) {
		return new Buffer(width, height);
	}
	
	public void bufferClear(Color color, float alpha) {
		Gdx.gl.glClearColor(color.r, color.g, color.b, alpha);
		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
	}
	
	public void bufferBegin(Buffer buffer) {
		tempBuffer = buffer;
		batch.end();
		this.buffer.end();
		tempBuffer.getBuffer().begin();
		/* рисовать на буффере с его проекцией */
		batch.setProjectionMatrix(buffer.projection);
		batch.begin();
		batch.setBlendFunction(-1, -1);
		Gdx.gl20.glBlendFuncSeparate(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA, GL20.GL_ONE, GL20.GL_ONE_MINUS_SRC_ALPHA);
	}

	public void bufferEnd() {
		batch.setBlendFunction(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA);
		tempBuffer.getBuffer().end();
		batch.end();
		batch.setProjectionMatrix(camera.combined);
		buffer.begin();
		batch.begin();
	}

	public void bufferDraw(Buffer buffer, float x, float y) {
		batch.setBlendFunction(GL20.GL_ONE, GL20.GL_ONE_MINUS_SRC_ALPHA);
		batch.draw(buffer.getBuffer().getColorBufferTexture(), x, y);
		batch.setBlendFunction(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA);
	}
	
	public void setAlpha(float alpha) {
		Color color = batch.getColor();
		batch.setColor(color.r, color.g, color.b, SimpleMath.clamp(alpha, 0, 1));
	}
	
	public void bufferDraw(Buffer buffer, float x, float y, float alpha) {
		Color color = batch.getColor();
		batch.setColor(color.r, color.g, color.b, SimpleMath.clamp(alpha, 0, 1));
		batch.setBlendFunction(GL20.GL_ONE, GL20.GL_ONE_MINUS_SRC_ALPHA);
		batch.draw(buffer.getBuffer().getColorBufferTexture(), x, y);
		batch.setBlendFunction(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA);
		batch.setColor(color.r, color.g, color.b, 1);
	}
	
	public void bufferDraw(Buffer buffer, float x, float y, float width, float height) {
		batch.setBlendFunction(GL20.GL_ONE, GL20.GL_ONE_MINUS_SRC_ALPHA);
		batch.draw(buffer.getBuffer().getColorBufferTexture(), x, y, width, height);
		batch.setBlendFunction(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA);
	}
	
	public void bufferDraw(Buffer buffer, float x, float y, float width, float height, float alpha) {
		Color color = batch.getColor();
		batch.setColor(color.r, color.g, color.b, SimpleMath.clamp(alpha, 0, 1));
		batch.setBlendFunction(GL20.GL_ONE, GL20.GL_ONE_MINUS_SRC_ALPHA);
		batch.draw(buffer.getBuffer().getColorBufferTexture(), x, y, width, height);
		batch.setBlendFunction(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA);
		batch.getColor().a = 1;
		batch.setColor(color.r, color.g, color.b, 1);
	}
	
	public void bufferDraw(Buffer buffer, float x, float y, boolean flipX, boolean flipY) {
		Texture texture = buffer.getBuffer().getColorBufferTexture();
		batch.setBlendFunction(GL20.GL_ONE, GL20.GL_ONE_MINUS_SRC_ALPHA);
		batch.draw(texture, x, y, texture.getWidth(), texture.getHeight(), 0, 0, texture.getWidth(), texture.getHeight(), flipX, flipY);
		batch.setBlendFunction(GL20.GL_SRC_ALPHA, GL20.GL_ONE_MINUS_SRC_ALPHA);
	}

	public Render(Batch batch, ShapeRenderer shapeRenderer, FrameBuffer buffer, OrthographicCamera camera) {
		this.batch = batch;
		defaultShader = batch.getShader();
		ShaderProgram.pedantic = false;
		this.shapeRenderer = shapeRenderer;
		this.buffer = buffer;
		this.camera = camera;
	}

}
