package blueberry.engine.json;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import blueberry.engine.IO.InputOutputManager;

public class JsonManager {
	private static GsonBuilder gsonBuilder;
	private static Gson gson;
			
	static {
		gsonBuilder = new GsonBuilder();
		gsonBuilder.excludeFieldsWithoutExposeAnnotation();
		gson = gsonBuilder.setPrettyPrinting().create();
	}
	
	public static String toJson(Object src) {
		return gson.toJson(src);
	}
	
	public static <T> T fromJson(String text, Class<T> classOfT) {
		return gson.fromJson(text, classOfT);
	}
	
	public static <T> T fromJsonFile(String path, Class<T> classOfT) {
		return gson.fromJson(InputOutputManager.read(path), classOfT);
	}
	
	public static boolean toJsonFile(String path, Object src) {
		return InputOutputManager.write(path, gson.toJson(src));
	}
}
