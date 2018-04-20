package blueberry.engine.render;

import java.awt.Image;
import java.awt.Toolkit;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import javax.swing.ImageIcon;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.files.FileHandle;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.PixmapIO;
import com.badlogic.gdx.utils.BufferUtils;
import com.badlogic.gdx.utils.ScreenUtils;

/*
 * Code from http://omtlab.com/java-store-image-in-clipboard/
 */

public class Screenshot {

	public static Screenshot data = new Screenshot();

	public void take() {

		byte[] pixels = ScreenUtils.getFrameBufferPixels(0, 0, Gdx.graphics.getBackBufferWidth(), Gdx.graphics.getBackBufferHeight(), true);

		for (int i = 4; i < pixels.length; i += 4) {
			pixels[i - 1] = (byte) 255;
		}

		Pixmap pixmap = new Pixmap(Gdx.graphics.getBackBufferWidth(), Gdx.graphics.getBackBufferHeight(), Pixmap.Format.RGBA8888);
		BufferUtils.copy(pixels, 0, pixmap.getPixels(), pixels.length);
		DateFormat dateFormat = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
		Date date = new Date();
		PixmapIO.writePNG(new FileHandle(System.getProperty("user.dir") + "/screenshot" + dateFormat.format(date) + ".png"), pixmap);
		pixmap.dispose();
		
		//ImageSelection imgSel = new ImageSelection(new ImageIcon(System.getProperty("user.dir") + "/screenshotGame.png").getImage());
	    //Toolkit.getDefaultToolkit().getSystemClipboard().setContents(imgSel, null);
	    
		//file.delete();
		
	}

	private Screenshot() {

	}

	class ImageSelection implements Transferable {
		  private Image image;

		  public ImageSelection(Image image) {
		    this.image = image;
		  }

		  public DataFlavor[] getTransferDataFlavors() {
		    return new DataFlavor[] { DataFlavor.imageFlavor };
		  }

		  public boolean isDataFlavorSupported(DataFlavor flavor) {
		    return DataFlavor.imageFlavor.equals(flavor);
		  }

		  public Object getTransferData(DataFlavor flavor)
		      throws UnsupportedFlavorException, IOException {
		    if (!DataFlavor.imageFlavor.equals(flavor)) {
		      throw new UnsupportedFlavorException(flavor);
		    }
		    return image;
		  }
		}

}
