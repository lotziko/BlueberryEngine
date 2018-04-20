package blueberry.engine.render;

import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Pixmap.Format;
import com.badlogic.gdx.graphics.Texture.TextureFilter;
import com.badlogic.gdx.graphics.glutils.FrameBuffer;
import com.badlogic.gdx.math.Matrix4;

public class Buffer {

	private FrameBuffer buffer;
	public int width, height;
	public Matrix4 projection;
	
	public FrameBuffer getBuffer() {
		return buffer;
	}
	
	public Buffer(int width, int height) {
		this.width = width;
		this.height = height;
		buffer = new FrameBuffer(Format.RGBA8888, width, height, false);
		buffer.getColorBufferTexture().setFilter(TextureFilter.Nearest, TextureFilter.Nearest);
		OrthographicCamera camera = new OrthographicCamera();
		camera.setToOrtho(true, width, height);
		projection = camera.combined;
	}
}
