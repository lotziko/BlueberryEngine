package blueberry.engine.IO;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;

public class InputOutputManager {
	
	public static String read(String path) {
		File file = new File(path);
		try {
			return new String(Files.readAllBytes(file.toPath()));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return "";
	}
	
	public static boolean write(String path, String text) {
		File file = new File(path);
		try {
			String[] lines = text.split("\n");
			PrintWriter pWriter = new PrintWriter(file);
			for(int i = 0; i < lines.length; i++) {
				pWriter.println(lines[i]);
			}
			pWriter.close();
			return true;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return false;
	}
	
	public static boolean write(String path, byte[] bytes) {
		File file = new File(path);
		try {
			PrintWriter pWriter = new PrintWriter(file);
			for(int i = 0; i < bytes.length; i++) {
				pWriter.print(bytes[i]);
			}
			pWriter.close();
			return true;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return false;
	}
}
