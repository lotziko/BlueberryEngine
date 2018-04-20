package blueberry.engine.objects;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class ObjectManager {
	
	private static ObjectManager manager = new ObjectManager();
	
	private ConcurrentHashMap<Integer, Object> objectList = new ConcurrentHashMap<>();
	private List<Object> renderOrder = new CopyOnWriteArrayList<>();
	private int IdCounter = 0;
	
	public static <T extends Object> T createObject(int x, int y, Class<T> type) {
		try {
			T instance = type.newInstance();
			instance.setID(manager.IdCounter);
			instance.setX(x);
			instance.setY(y);
			instance.create();
			manager.objectList.put(manager.IdCounter, instance);
			manager.renderOrder.add(instance);
			manager.renderOrder.sort(new Comparator<Object>() {

				@Override
				public int compare(Object o1, Object o2) {
					return Integer.compare(o1.depth, o2.depth);
				}
			});
			++manager.IdCounter;
			return instance;
		} catch (InstantiationException | IllegalAccessException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static <T extends Object> T createObject(int x, int y, Class<T> type, T object) {
		object.setID(manager.IdCounter);
		object.setX(x);
		object.setY(y);
		manager.objectList.put(manager.IdCounter, object);
		manager.renderOrder.add(object);
		manager.renderOrder.sort(new Comparator<Object>() {

			@Override
			public int compare(Object o1, Object o2) {
				return Integer.compare(o1.depth, o2.depth);
			}
		});
		++manager.IdCounter;
		return object;
	}
	
	public static void destroyObject(int ID) {
		manager.objectList.remove(ID);
		manager.renderOrder.removeIf(obj -> obj.ID == ID);
	}
	
	public static void destroyObject(Object object) {
		if (object != null) {
			object.destroy();
			manager.objectList.values().remove(object);
			manager.renderOrder.remove(object);
		}
	}
	
	/*public static<T extends Object> List<Object> getInstancesOfClass(Class<T> type) {
		for (Object object : manager.objectList.values()) {
			if (object instanceof T) {
				
			}
		}
		return null;
	}*/
	
	public static void step() {
		for (Object object : manager.objectList.values()) {
			object.step();
		}
	}
	
	public static void drawBegin() {
		for (Object object : manager.renderOrder) {
			object.drawBegin();
		}
	}
	
	public static void draw() {
		for (Object object : manager.renderOrder) {
			object.draw();
		}
	}
	
	public static void drawEnd() {
		for (Object object : manager.renderOrder) {
			object.drawEnd();
		}
	}
	
	public static void drawGUI() {
		for (Object object : manager.renderOrder) {
			object.drawGUI();
		}
	}
	
	public static void clear() {
		for (Object object : manager.objectList.values()) {
			destroyObject(object);
		}
		manager.objectList.clear();
		manager.renderOrder.clear();
	}
	
}
