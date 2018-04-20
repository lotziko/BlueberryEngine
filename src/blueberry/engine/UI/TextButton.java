package blueberry.engine.UI;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.GlyphLayout;
import com.badlogic.gdx.graphics.g2d.NinePatch;

public class TextButton extends Button {

	private BitmapFont font;
	private String text;
	private float textPosX, textPosY;
	private Color textColor = Color.WHITE;
	private float alpha = 1f;

	public String getText() {
		return text;
	}
	
	@Override
	public void setPosition(float x, float y) {
		super.setPosition(x, y);
		GlyphLayout layout = new GlyphLayout(font, text);
		textPosX = this.x + (width - layout.width) * 0.5f;
		textPosY = this.y + (height - layout.height) * 0.5f;
	}

	public void setText(String text) {
		this.text = text;
		GlyphLayout layout = new GlyphLayout(font, text);
		textPosX = x + (width - layout.width) * 0.5f;
		textPosY = y + (height - layout.height) * 0.5f;
	}
	
	public Color getTextColor() {
		return textColor;
	}

	public void setTextColor(Color textColor) {
		this.textColor = textColor;
	}
	
	public float getAlpha() {
		return alpha;
	}

	public void setAlpha(float alpha) {
		this.alpha = alpha;
	}

	@Override
	public void draw(Batch batch) {
		super.draw(batch);
		if (isVisible) {
			if (font != null) {
				Color color = new Color(font.getColor());
				color.a = alpha;
				font.setColor(textColor);
				font.draw(batch, text, textPosX, textPosY);
				font.setColor(color);
			}
		}
	}
	
	public TextButton(float x, float y, float width, float height, NinePatch background, NinePatch hover, BitmapFont font, String text, int align) {
		super(x, y, width, height, background, hover, align);
		this.font = font;
		setText(text);
	}
	
}
